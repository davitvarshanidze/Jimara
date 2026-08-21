#include "SceneAccelerationStructures.h"
#include "../../../GraphicsSimulation/GraphicsSimulation.h"


namespace Jimara {
	struct SceneAccelerationStructures::Helpers {
		using DirtyType_t = std::underlying_type_t<DirtyType>;

		inline static void Build(
			Graphics::CommandBuffer* commandBuffer, 
			const BlasDesc& desc, 
			Graphics::BottomLevelAccelerationStructure* blas,
			Reference<Graphics::BottomLevelAccelerationStructure>& baseBlas,
			std::atomic<DirtyType_t>& dirtyType,
			std::atomic_bool& initialized) {
			
			desc.displacementJob(commandBuffer, desc.displacementJobId);
			
			const bool wasBuilt = initialized.exchange(true);
			const bool rebuildRequested = ((dirtyType.exchange((DirtyType_t)DirtyType::NONE) & ((DirtyType_t)DirtyType::NEEDS_REBUILD)) != 0);

			const bool refit = (
				wasBuilt && (!rebuildRequested) &&
				((desc.flags & Flags::REFIT_ON_REBUILD) != Flags::NONE));

			Graphics::BottomLevelAccelerationStructure* const baseAc =
				(baseBlas != nullptr) ? baseBlas.operator->() :
				refit ? blas :
				nullptr;
			
			blas->Build(commandBuffer,
				desc.vertexBuffer, desc.vertexStride, desc.vertexPositionOffset,
				desc.indexBuffer,
				baseAc,
				desc.vertexCount,
				desc.faceCount * 3u,
				desc.indexOffset);

			if (baseBlas != nullptr && (desc.flags & Flags::DO_NOT_USE_BASE_BLAS_AFTER_FIRST_BUILD) != Flags::NONE)
				baseBlas = nullptr;
		}

		struct BuildCommand : public virtual BulkAllocated {
			const BlasDesc desc;
			Reference<Graphics::BottomLevelAccelerationStructure> blas;

			mutable Reference<Graphics::BottomLevelAccelerationStructure> variantBaseBlas;
			bool isVariant = false;

			mutable std::atomic<DirtyType_t> dirtyType = (DirtyType_t)DirtyType::NONE;

			mutable std::atomic_bool initialized = false;

			inline BuildCommand(const BlasDesc& descriptor,
				Graphics::BottomLevelAccelerationStructure* as)
				: desc(descriptor), blas(as) {
				assert(blas != nullptr);
			}
			inline virtual ~BuildCommand() {}
		};

		struct DependencyCollector : public virtual Object {
			EventInstance<Callback<JobSystem::Job*>> collectionEvents;
		};

#pragma warning(disable: 4250)
		class Queues : public virtual JobSystem::Job {
		private:
			Reference<SceneContext> m_context;
			const Reference<GraphicsSimulation::JobDependencies> m_simulationJobs;
			const Reference<const DependencyCollector> m_dependencies;

			std::mutex m_commandPoolLock;
			Reference<Graphics::CommandPool> m_commandPool;
			std::queue<std::pair<Reference<Graphics::PrimaryCommandBuffer>, uint64_t>> m_runninngBuildCommands;

			using OneTimeBuildList = std::vector<Reference<const Object>>;
			using OneTimeBuildQueue = std::vector<OneTimeBuildList>;
			struct BuildBatch {
				std::mutex oneTimeBuildLock;
				std::mutex oneTimeBuildListLock;
				uint8_t oneTimeCommandListFrontBuffer = static_cast<uint8_t>(0u);
				OneTimeBuildQueue oneTimeBuildCommands[2u];

				std::mutex perFrameBuildLock;
				std::mutex perFrameBuildCommandSynchLock;
				DelayedObjectSet<const BuildCommand> perFrameBuildCommands;
			};
			BuildBatch m_baseBuildBatch;
			BuildBatch m_variantBuildBatch;

			std::atomic_uint64_t m_lastBuildFrame;


			inline void PerformOneTimeBuild(Graphics::CommandBuffer* commands, BuildBatch* batch) {
				std::unique_lock<decltype(batch->oneTimeBuildLock)> lock(batch->oneTimeBuildLock);

				// Swap buffers and obtain back buffer:
				OneTimeBuildQueue* oneTimeBuildCommands = nullptr;
				{
					std::unique_lock<decltype(batch->oneTimeBuildListLock)> lock(batch->oneTimeBuildListLock);
					oneTimeBuildCommands = batch->oneTimeBuildCommands + static_cast<size_t>(batch->oneTimeCommandListFrontBuffer);
					batch->oneTimeCommandListFrontBuffer ^= static_cast<uint8_t>(1u);
				}

				// Build blas instances:
				for (size_t listId = 0u; listId < oneTimeBuildCommands->size(); listId++) {
					const OneTimeBuildList& buildList = oneTimeBuildCommands->data()[listId];
					const auto end = buildList.data() + buildList.size();
					for (auto it = buildList.data(); it < end; it++) {
						const BuildCommand* command = dynamic_cast<const BuildCommand*>(it->operator->());
						assert(command != nullptr);
						assert(command->isVariant == (batch == &m_variantBuildBatch));
						Build(commands, command->desc, command->blas,
							command->variantBaseBlas,
							command->dirtyType,
							command->initialized);
					}
				}

				// Clear back-buffer:
				oneTimeBuildCommands->clear();
			}

			inline void PerformPerFrameRebuilds(Graphics::CommandBuffer* commands, BuildBatch* batch) {
				std::unique_lock<decltype(batch->perFrameBuildLock)> lock(batch->perFrameBuildLock);

				// Flush:
				{
					std::unique_lock<decltype(batch->perFrameBuildCommandSynchLock)> lock(batch->perFrameBuildCommandSynchLock);
					batch->perFrameBuildCommands.Flush([&](const auto&...) {}, [&](const auto&...) {});
				}

				// [Re] Build:
				const Reference<const BuildCommand>* ptr = batch->perFrameBuildCommands.Data();
				const Reference<const BuildCommand>* const end = ptr + batch->perFrameBuildCommands.Size();
				while (ptr < end) {
					assert((*ptr)->isVariant == (batch == &m_variantBuildBatch));
					Build(commands, (*ptr)->desc, (*ptr)->blas, 
						(*ptr)->variantBaseBlas,
						(*ptr)->dirtyType,
						(*ptr)->initialized);
					ptr++;
				}
			}

			inline void CleanRunningBuildCommands(const std::unique_lock<decltype(Queues::m_commandPoolLock)>&)
			{
				const uint64_t frameIndex = m_context->FrameIndex();
				const uint64_t maxInFlightCommandCount = m_context->Graphics()->Configuration().MaxInFlightCommandBufferCount();
				while (!m_runninngBuildCommands.empty()) {
					const std::pair<Reference<Graphics::PrimaryCommandBuffer>, uint64_t> entry = m_runninngBuildCommands.front();
					const uint64_t frameDistance = (frameIndex - entry.second);
					if (frameDistance >= maxInFlightCommandCount || // Frame index large enough to wait and discard [will work even in case of an overflow].
						m_runninngBuildCommands.size() >= MAX_RUNNING_COMMAND_BUFFERS) {
						entry.first->Wait();
						m_runninngBuildCommands.pop();
					}
					else break;
				}
			}

		protected:
			virtual void Execute() {
				// Protect against double execution:
				{
					const uint64_t frameId = m_context->FrameIndex();
					if (m_lastBuildFrame.exchange(frameId) == frameId)
						return;
				}

				// Obtain command buffer:
				const Graphics::InFlightBufferInfo buffer = m_context->Graphics()->GetWorkerThreadCommandBuffer();
				if (buffer.commandBuffer == nullptr)
					return;

				// Build stuff:
				PerformOneTimeBuild(buffer, &m_baseBuildBatch);
				PerformPerFrameRebuilds(buffer, &m_baseBuildBatch);
				PerformOneTimeBuild(buffer, &m_variantBuildBatch);
				PerformPerFrameRebuilds(buffer, &m_variantBuildBatch);
				CleanRunningBuildCommands(std::unique_lock<decltype(Queues::m_commandPoolLock)>(m_commandPoolLock));
			}

			virtual void CollectDependencies(Callback<Job*> addDependency) {
				m_simulationJobs->CollectDependencies(addDependency);
				m_dependencies->collectionEvents(addDependency);
			}

		public:
			inline Queues(SceneContext* context, 
				GraphicsSimulation::JobDependencies* simulationJobs, 
				const DependencyCollector* dependencies, 
				Graphics::CommandPool* pool)
				: m_context(context)
				, m_simulationJobs(simulationJobs)
				, m_dependencies(dependencies)
				, m_commandPool(pool) {
				assert(m_context != nullptr);
				assert(m_simulationJobs != nullptr);
				assert(m_dependencies != nullptr);
				assert(m_commandPool != nullptr);
				m_lastBuildFrame = context->FrameIndex() - 1u;
			}

			inline virtual ~Queues() {}

			inline SceneContext* Context()const { return m_context; }

			static const constexpr std::size_t MAX_RUNNING_COMMAND_BUFFERS = 1024u;

			class OneTimeCommandBuffer {
			private:
				const std::unique_lock<decltype(Queues::m_commandPoolLock)> m_lock;
				const Reference<Queues> m_queues;
				const Reference<Graphics::PrimaryCommandBuffer> m_commandBuffer = nullptr;

			public:
#pragma warning(disable: 26115)
				OneTimeCommandBuffer(Queues* queues)
					: m_lock(queues->m_commandPoolLock)
					, m_queues(queues)
					, m_commandBuffer(queues->m_commandPool->CreatePrimaryCommandBuffer()) {
					if (m_commandBuffer != nullptr)
						m_commandBuffer->BeginRecording();
				}
#pragma warning(default: 26115)

				~OneTimeCommandBuffer() {
					if (m_commandBuffer != nullptr) {
						m_commandBuffer->EndRecording();
						m_queues->m_context->Graphics()->Device()->GraphicsQueue()->ExecuteCommandBuffer(m_commandBuffer);
						m_queues->CleanRunningBuildCommands(m_lock);
						m_queues->m_runninngBuildCommands.push(std::make_pair(m_commandBuffer, m_queues->m_context->FrameIndex()));
					}
				}

				Graphics::PrimaryCommandBuffer* Buffer()const { return m_commandBuffer; }
			};

			inline void ScheduleOneTimeBuild(const BuildCommand* command) {
				BuildBatch* batch = command->isVariant ? &m_variantBuildBatch : &m_baseBuildBatch;
				std::unique_lock<decltype(batch->oneTimeBuildListLock)> lock(batch->oneTimeBuildListLock);
				OneTimeBuildQueue& queue = batch->oneTimeBuildCommands[batch->oneTimeCommandListFrontBuffer];
				if (queue.empty())
					queue.push_back({});
				OneTimeBuildList& list = queue.back();
				list.push_back(command);
			}

			inline void ScheduleOneTimeBuild(std::vector<Reference<const Object>>&& list, bool variant) {
				BuildBatch* batch = variant ? &m_variantBuildBatch : &m_baseBuildBatch;
				if (list.size() <= 0u)
					return;
				std::unique_lock<decltype(batch->oneTimeBuildListLock)> lock(batch->oneTimeBuildListLock);
				OneTimeBuildQueue& queue = batch->oneTimeBuildCommands[batch->oneTimeCommandListFrontBuffer];
				OneTimeBuildList& lst = queue.emplace_back();
				assert(lst.size() == 0u);
				std::swap(list, lst);
				assert(lst.size() > 0u);
				assert(list.size() == 0u);
			}

			inline void AddPerFrameBuild(const BuildCommand* command) {
				BuildBatch* batch = command->isVariant ? &m_variantBuildBatch : &m_baseBuildBatch;
				std::unique_lock<decltype(batch->perFrameBuildCommandSynchLock)> lock(batch->perFrameBuildCommandSynchLock);
				batch->perFrameBuildCommands.ScheduleAdd(command);
			}

			inline void RemovePerFrameBuild(const BuildCommand* command) {
				BuildBatch* batch = command->isVariant ? &m_variantBuildBatch : &m_baseBuildBatch;
				std::unique_lock<decltype(batch->perFrameBuildCommandSynchLock)> lock(batch->perFrameBuildCommandSynchLock);
				batch->perFrameBuildCommands.ScheduleRemove(command);
			}
		};

		struct BlasInstance final : public virtual Blas, public virtual ObjectCache<BlasDesc>::StoredObject {
			Reference<Queues> queues;
			Reference<BuildCommand> buildCommand;

			inline BlasInstance(const BlasDesc& desc, Graphics::BottomLevelAccelerationStructure* as)
				: buildCommand(Object::Instantiate<BuildCommand>(desc, as)) {
				assert(buildCommand != nullptr);
				assert(!buildCommand->initialized.load());
				assert(buildCommand->blas != nullptr);
			}

			inline virtual ~BlasInstance() {
				buildCommand->initialized.store(true);
				if (queues != nullptr) {
					assert(buildCommand != nullptr);
					queues->RemovePerFrameBuild(buildCommand);
					queues = nullptr;
				}
				buildCommand = nullptr;
			}

			inline static Reference<BlasInstance> Create(const BlasDesc& desc, SceneContext* context, Queues* queues, Graphics::BottomLevelAccelerationStructure* baseBlas) {
				auto fail = [&](const auto... message) {
					context->Log()->Error("SceneAccelerationStructures::Helpers::BlasInstance::Create - ", message...);
					return nullptr;
				};

				if (desc.vertexBuffer == nullptr)
					return fail("Vertex buffer missing! [File: ", __FILE__, "; Line: ", __LINE__, "]");

				// Create AS instance:
				Graphics::BottomLevelAccelerationStructure::Properties asProps = {};
				{
					asProps.maxTriangleCount = desc.faceCount;
					asProps.maxVertexCount = desc.vertexCount;
					asProps.vertexFormat = desc.vertexFormat;
					asProps.indexFormat = desc.indexFormat;
					asProps.flags = decltype(asProps.flags)::NONE;
					if ((desc.flags & Flags::REBUILD_ON_EACH_FRAME) != Flags::NONE)
						asProps.flags |= decltype(asProps.flags)::ALLOW_UPDATES;
					if ((desc.flags & Flags::PREFER_FAST_BUILD) != Flags::NONE)
						asProps.flags |= decltype(asProps.flags)::PREFER_FAST_BUILD;
					if ((desc.flags & Flags::PREVENT_DUPLICATE_ANY_HIT_INVOCATIONS) != Flags::NONE)
						asProps.flags |= decltype(asProps.flags)::PREVENT_DUPLICATE_ANY_HIT_INVOCATIONS;
				}
				const Reference<Graphics::BottomLevelAccelerationStructure> as = context->Graphics()->Device()->CreateBottomLevelAccelerationStructure(asProps);
				if (as == nullptr)
					return fail("Failed to create Acceleration structure instance! [File: ", __FILE__, "; Line: ", __LINE__, "]");

				// Create instance:
				const Reference<BlasInstance> instance = Object::Instantiate<BlasInstance>(desc, as);
				assert(instance->buildCommand != nullptr);
				assert(!instance->buildCommand->initialized.load());

				// For blas-variants, we need to set base-blas:
				instance->buildCommand->variantBaseBlas = baseBlas;
				instance->buildCommand->isVariant = (instance->buildCommand->variantBaseBlas != nullptr);

				// For dirty-queues, we always need a 'live' build command:
				assert(instance->buildCommand != nullptr);
				instance->buildCommand->dirtyType = (DirtyType_t)DirtyType::NEEDS_REBUILD;

				// Optionally build AS if the request is urgent:
				if ((desc.flags & Flags::INITIAL_BUILD_SCHEDULE_URGENT) != Flags::NONE) {
					Queues::OneTimeCommandBuffer commands(queues);
					if (commands.Buffer() == nullptr)
						fail("Failed to create command buffer! [File: ", __FILE__, "; Line: ", __LINE__, "]");
					else {
						Build(commands.Buffer(), desc, instance->buildCommand->blas,
							instance->buildCommand->variantBaseBlas,
							instance->buildCommand->dirtyType,
							instance->buildCommand->initialized);
						assert(instance->buildCommand->initialized.load());
					}
				}

				// If urgent build was not requered and one-time build is enough, schedule one-time build:
				if ((desc.flags & Flags::INITIAL_BUILD_SCHEDULE_URGENT) == Flags::NONE &&
					(desc.flags & Flags::REBUILD_ON_EACH_FRAME) == Flags::NONE)
					queues->ScheduleOneTimeBuild(instance->buildCommand);

				// If per-frame-rebuild is required, we add to the per-frame build jobs:
				if ((desc.flags & Flags::REBUILD_ON_EACH_FRAME) != Flags::NONE) {
					instance->queues = queues;
					queues->AddPerFrameBuild(instance->buildCommand);
				}
				else assert(instance->queues == nullptr);

				// Done:
				return instance;
			}

			inline static const BlasInstance* Self(const Blas* selfPtr) {
				return dynamic_cast<const BlasInstance*>(selfPtr);
			}
		};

		class BlasCache final : public virtual ObjectCache<BlasDesc> {
		public:
			inline Reference<BlasInstance> GetInstance(BlasDesc desc, SceneContext* context, Queues* queues, Graphics::BottomLevelAccelerationStructure* baseBlas) {
				if (baseBlas == nullptr)
					desc.flags &= ~Flags::DO_NOT_USE_BASE_BLAS_AFTER_FIRST_BUILD;
				return GetCachedOrCreate(desc, [&]() { return BlasInstance::Create(desc, context, queues, baseBlas); });
			}
		};

		class ScheduledJob : public virtual JobSystem::Job {
		private:
			const Reference<Queues> queues;

		public:
			inline ScheduledJob(Queues* q) : queues(q) {}
			inline virtual ~ScheduledJob() {}

		protected:
			virtual void Execute() {}

			virtual void CollectDependencies(Callback<Job*> addDependency) { addDependency(queues); }
		};

		struct Instance final : public virtual SceneAccelerationStructures, public virtual ObjectCache<Reference<SceneContext>>::StoredObject {
			const Reference<Queues> queues;
			const Reference<BlasCache> cache = Object::Instantiate<BlasCache>();
			const Reference<BlasCache> variantCache = Object::Instantiate<BlasCache>();
			const Reference<DependencyCollector> dependencyCollector;
			const Reference<ScheduledJob> job;

			inline Instance(Queues* updateJob, DependencyCollector* dependencies)
				: queues(updateJob), dependencyCollector(dependencies)
				, job(Object::Instantiate<ScheduledJob>(updateJob)) {
				assert(queues != nullptr);
				assert(dependencyCollector != nullptr);
				queues->Context()->Graphics()->RenderJobs().Add(job);
			}

			virtual inline ~Instance() {
				queues->Context()->Graphics()->RenderJobs().Remove(job);
			}

			inline static Reference<Instance> Create(SceneContext* context) {
				if (context == nullptr)
					return nullptr;

				auto fail = [&](const auto... message) {
					context->Log()->Error("SceneAccelerationStructures::Helpers::Instance::Create - ", message...);
					return nullptr;
				};
				const Reference<GraphicsSimulation::JobDependencies> simulationJobs = GraphicsSimulation::JobDependencies::For(context);
				if (simulationJobs == nullptr)
					return fail("Failed to get simulation job dependencies! [File: ", __FILE__, "; Line: ", __LINE__, "]");

				const Reference<DependencyCollector> dependencyCollector = Object::Instantiate<DependencyCollector>();

				const Reference<Graphics::CommandPool> commandPool = context->Graphics()->Device()->GraphicsQueue()->CreateCommandPool();
				if (commandPool == nullptr)
					return fail("Failed to create command pool! [File: ", __FILE__, "; Line: ", __LINE__, "]");

				Reference<Queues> queues = Object::Instantiate<Queues>(context, simulationJobs, dependencyCollector, commandPool);

				return Object::Instantiate<Instance>(queues, dependencyCollector);
			}

			inline static Instance* Self(SceneAccelerationStructures* selfPtr) {
				return dynamic_cast<Instance*>(selfPtr);
			}
		};

		struct InstanceCache final : public virtual ObjectCache<Reference<SceneContext>> {
			static Reference<Instance> Get(SceneContext* context) {
				static InstanceCache cache;
				return cache.GetCachedOrCreate(context, [&]() { return Instance::Create(context); });
			}
		};
#pragma warning(default: 4250)
	};

	SceneAccelerationStructures::SceneAccelerationStructures() {}

	SceneAccelerationStructures::~SceneAccelerationStructures() {}

	Reference<SceneAccelerationStructures> SceneAccelerationStructures::Get(SceneContext* context) {
		if (context == nullptr)
			return nullptr;
		if (!context->Graphics()->Device()->PhysicalDevice()->HasFeatures(Graphics::PhysicalDevice::DeviceFeatures::RAY_TRACING)) {
			context->Log()->Error("SceneAccelerationStructures::Get - ",
				"Graphics device does not support hardware Ray-Tracing features! [File: ", __FILE__, "; Line: ", __LINE__, "]");
			return nullptr;
		}
		else return Helpers::InstanceCache::Get(context);
	}

	Reference<SceneAccelerationStructures::Blas> SceneAccelerationStructures::GetBlas(BlasDesc& desc) {
		Helpers::Instance* const self = Helpers::Instance::Self(this);
		return self->cache->GetInstance(desc, self->ObjectCacheKey(), self->queues, nullptr);
	}

#if JIMARA_SceneAccelerationStructures_ENABLE_BlasVariant
	Reference<SceneAccelerationStructures::Blas> SceneAccelerationStructures::GetBlas(VariantDesc& desc) {
		Helpers::Instance* const self = Helpers::Instance::Self(this);
		assert(self != nullptr);
		if (desc.baseBlas == nullptr) {
			self->ObjectCacheKey()->Log()->Error(
				"SceneAccelerationStructures::GetBlas - Base blas not provided!",
				"[File: ", __FILE__, "; Line: ", __LINE__, "]");
			return nullptr;
		}
		const Helpers::BlasInstance* const base = Helpers::BlasInstance::Self(desc.baseBlas);
		assert(base != nullptr);
		if (base->buildCommand->isVariant) {
			self->ObjectCacheKey()->Log()->Error(
				"SceneAccelerationStructures::GetBlas - Base blas should not be a blas-variant!",
				"[File: ", __FILE__, "; Line: ", __LINE__, "]");
			return nullptr;
		}
		BlasDesc blasDesc = {};
		{
			blasDesc.vertexBuffer = desc.vertexBuffer;
			blasDesc.indexBuffer = desc.baseBlas->Descriptor().indexBuffer;
			blasDesc.vertexFormat = desc.baseBlas->Descriptor().vertexFormat;
			blasDesc.indexFormat = desc.baseBlas->Descriptor().indexFormat;
			blasDesc.vertexPositionOffset = desc.vertexPositionOffset;
			blasDesc.vertexStride = desc.vertexStride;
			blasDesc.vertexCount = desc.baseBlas->Descriptor().vertexCount;
			blasDesc.faceCount = desc.baseBlas->Descriptor().faceCount;
			blasDesc.indexOffset = desc.baseBlas->Descriptor().indexOffset;
			blasDesc.flags = desc.flags;
			blasDesc.displacementJob = desc.displacementJob;
			blasDesc.displacementJobId = desc.displacementJobId;
		}
		return self->variantCache->GetInstance(
			blasDesc, self->ObjectCacheKey(), self->queues, base->buildCommand->blas);
	}
#endif

	Event<Callback<JobSystem::Job*>>& SceneAccelerationStructures::OnCollectBuildDependencies() {
		return Helpers::Instance::Self(this)->dependencyCollector->collectionEvents;
	}

	void SceneAccelerationStructures::CollectBuildJobs(const Callback<JobSystem::Job*> report) {
		report(Helpers::Instance::Self(this)->queues);
	}

	const SceneAccelerationStructures::BlasDesc& SceneAccelerationStructures::Blas::Descriptor()const {
		return Helpers::BlasInstance::Self(this)->ObjectCacheKey();
	}

	Graphics::BottomLevelAccelerationStructure* SceneAccelerationStructures::Blas::AcccelerationStructure()const {
		const Helpers::BlasInstance* const self = Helpers::BlasInstance::Self(this);
		assert(self != nullptr);
		const Helpers::BuildCommand* buildCommand = self->buildCommand;
		assert(buildCommand != nullptr);
		return buildCommand->initialized.load() ? buildCommand->blas.operator->() : nullptr;
	}


#if JIMARA_SceneAccelerationStructures_ENABLE_DirtyQueue
	SceneAccelerationStructures::DirtyQueue::DirtyQueue(SceneAccelerationStructures* set) : m_set(set) {
		assert(m_set != nullptr);
	}

	SceneAccelerationStructures::DirtyQueue::~DirtyQueue() {
		const Helpers::Instance* instance = Helpers::Instance::Self(m_set);
		assert(instance != nullptr);
		if (!m_queue.empty()) {
			instance->queues->ScheduleOneTimeBuild(std::move(m_queue), false);
			assert(m_queue.empty());
		}
		if (!m_variantQueue.empty()) {
			instance->queues->ScheduleOneTimeBuild(std::move(m_variantQueue), true);
			assert(m_variantQueue.empty());
		}
	}

	void SceneAccelerationStructures::DirtyQueue::Submit(Blas* blas, DirtyType dirty) {
		if (blas == nullptr || dirty <= DirtyType::NONE)
			return;
		const Helpers::BlasInstance* const as = Helpers::BlasInstance::Self(blas);
		assert(as != nullptr);
		assert(as->buildCommand != nullptr);
		const DirtyType flags = (DirtyType)as->buildCommand->dirtyType.fetch_or((Helpers::DirtyType_t)dirty);
		// We check for queues, because if the blas is already updated on a per-frame basis, we don't have a need to enqueue:
		if (flags == DirtyType::NONE && as->queues == nullptr) {
			if (as->buildCommand->isVariant)
				m_variantQueue.push_back(as->buildCommand);
			else m_queue.push_back(as->buildCommand);
		}
	}
#endif
}
