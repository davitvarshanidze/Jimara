#include "TerrainRenderer.h"
#include "../../Data/Serialization/Helpers/SerializerMacros.h"
#include "../../Environment/Rendering/SceneObjects/Objects/GraphicsObjectDescriptor.h"
#include "../../Data/Materials/SampleDiffuse/SampleDiffuseShader.h"


namespace Jimara {
	struct TerrainRenderer::Helpers {
		// Each grid tile will be DEFAULT_TILE_GRID_SIZE x DEFAULT_TILE_GRID_SIZE by default.
		static const constexpr uint32_t DEFAULT_TILE_GRID_SIZE = 8u;

#pragma region Shared Index Buffer
		// Key for shring the index buffer accross all terrain instances that can share it.
		struct TileIndexBufferKey {
			Reference<Graphics::GraphicsDevice> device;
			uint32_t tileGridSize = DEFAULT_TILE_GRID_SIZE;

			inline bool operator==(const TileIndexBufferKey& other)const {
				return
					device == other.device &&
					tileGridSize == other.tileGridSize;
			}

			struct Hash {
				inline size_t operator()(const TileIndexBufferKey& key)const {
					return MergeHashes(
						std::hash<Graphics::GraphicsDevice*>()(key.device.operator->()),
						std::hash<uint32_t>()(key.tileGridSize));
				}
			};
		};


		// Index buffer for individual terrain tiles.
		struct TileIndexBuffer : public virtual ObjectCache<TileIndexBufferKey, TileIndexBufferKey::Hash>::StoredObject {
		private:
			const Graphics::ArrayBufferReference<uint32_t> m_buffer;
			
			struct Cache : public virtual ObjectCache<TileIndexBufferKey, TileIndexBufferKey::Hash> {
				template<typename GetFn>
				inline Reference<TileIndexBuffer> Get(const TileIndexBufferKey& key, const GetFn& createFn) { return GetCachedOrCreate(key, createFn); }
			};

			inline TileIndexBuffer(const Graphics::ArrayBufferReference<uint32_t>& buffer) : m_buffer(buffer) {
				assert(m_buffer != nullptr);
			}

		public:
			// Gets or creates a shared index buffer for all terrains.
			inline static Reference<TileIndexBuffer> Get(OS::Logger* log, Graphics::GraphicsDevice* device, uint32_t tileGridSize) {
				// Failure callback:
				const auto fail = [&](const auto&... message) {
					if (log != nullptr)
						log->Error("TerrainRenderer::Helpers::TileIndexBuffer::Get - ", message...);
					return nullptr;
				};
				
				// Check that device is valid:
				if (device == nullptr)
					return fail("Device not provided! [File: ", __FILE__, "; Line: ", __LINE__, "]");
				
				// Create key:
				TileIndexBufferKey key = {};
				key.device = device;
				key.tileGridSize = (tileGridSize > 0u) ? tileGridSize : DEFAULT_TILE_GRID_SIZE;

				// Return cached instance with creation function:
				static Cache cache;
				return cache.Get(key, [&]() -> Reference<TileIndexBuffer> {
					// Allocate buffer:
					const Graphics::ArrayBufferReference<uint32_t> buffer = key.device->CreateArrayBuffer<uint32_t>(
						key.tileGridSize * key.tileGridSize * 6u, Graphics::ArrayBuffer::CPUAccess::CPU_WRITE_ONLY);
					if (buffer == nullptr)
						return fail("Failed to allocate tile grid buffer! [File: ", __FILE__, "; Line: ", __LINE__, "]");

					// Map buffer:
					uint32_t* data = buffer.Map();
					if (data == nullptr)
						return fail("Failed to map tile grid buffer! [File: ", __FILE__, "; Line: ", __LINE__, "]");
					
					// For each rectangle within the grid, fill in indices of the two triangles making it up:
					for (uint32_t y = 0u; y < key.tileGridSize; y++)
						for (uint32_t x = 0u; x < key.tileGridSize; x++) {
							uint32_t* const base = data + ((y * key.tileGridSize + x) * 6u);
							
							// Vertex buffer grid will need to be ((tileGridSize + 1u) x (tileGridSize + 1u)), naturally, so we start at a:
							const uint32_t a = x + (y * (key.tileGridSize + 1u));
							const uint32_t b = a + 1u;
							const uint32_t c = b + key.tileGridSize;
							const uint32_t d = c + 1u;
							
							// 'Upper' face:
							base[0u] = a;
							base[1u] = b;
							base[2u] = c;

							// 'Lower' face:
							base[3u] = c;
							base[4u] = b;
							base[5u] = d;
						}
					
					// Done writing to the buffer:
					buffer->Unmap(true);

					// Create shared object:
					Reference<TileIndexBuffer> indexBuffer = new TileIndexBuffer(buffer);
					indexBuffer->ReleaseRef();
					assert(indexBuffer->RefCount() == 1u);
					return indexBuffer;
					});
			}

			virtual ~TileIndexBuffer() {}

			inline Graphics::GraphicsDevice* Device() { return ObjectCacheKey().device; }
			inline uint32_t TileGridSize() { return ObjectCacheKey().tileGridSize; }
			inline const Graphics::ArrayBufferReference<uint32_t>& Buffer() { return m_buffer; }
		};
#pragma endregion




		
#pragma region Vertex Buffer
		// Creates a tile vertex buffer;
		// <para/> This is temporary, so we don't care about caching as such, we just go ahead and create an instance per graphics object for now.
		inline static Graphics::ArrayBufferReference<MeshVertex> CreateVertexBuffer(OS::Logger* log, Graphics::GraphicsDevice* device, uint32_t tileGridSize)
		{
			if (tileGridSize <= 0u)
				tileGridSize = DEFAULT_TILE_GRID_SIZE;

			// Failure callback:
			const auto fail = [&](const auto&... message) {
				if (log != nullptr)
					log->Error("TerrainRenderer::Helpers::CreateVertexBuffer - ", message...);
				return nullptr;
			};

			// Check that device is valid:
			if (device == nullptr)
				return fail("Device not provided! [File: ", __FILE__, "; Line: ", __LINE__, "]");

			const Graphics::ArrayBufferReference<MeshVertex> buffer = device->CreateArrayBuffer<MeshVertex>(
				(tileGridSize + 1u) * (tileGridSize + 1u), Graphics::ArrayBuffer::CPUAccess::CPU_WRITE_ONLY);
			if (buffer == nullptr)
				return fail("Failed to allocate tile grid buffer! [File: ", __FILE__, "; Line: ", __LINE__, "]");

			// Map buffer:
			MeshVertex* data = buffer.Map();
			if (data == nullptr)
				return fail("Failed to map tile grid buffer! [File: ", __FILE__, "; Line: ", __LINE__, "]");

			// For each verex within the grid, fill in the details in the simplest possible manner:
			for (uint32_t y = 0u; y <= tileGridSize; y++)
				for (uint32_t x = 0u; x <= tileGridSize; x++) {
					MeshVertex& vertex = data[((tileGridSize + 1u) * y) + x];
					const float xF = static_cast<float>(x);
					const float yF = static_cast<float>(y);
					const float sizeF = static_cast<float>(tileGridSize);
					vertex.position = Vector3(xF - (sizeF * 0.5f), 0.0f, yF - (sizeF * 0.5f));
					vertex.normal = Vector3(0.0f, 1.0f, 0.0f);
					vertex.uv = Vector2(xF, yF) / sizeF;
				}

			// Done:
			buffer->Unmap(true);
			return buffer;
		}
#pragma endregion





#pragma region Graphics Object
		struct GraphicsObjectData : public virtual Object {
			Reference<Material::CachedInstance> cachedMaterialInstance;
			Reference<TileIndexBuffer> indexBuffer;
			Graphics::ArrayBufferReference<MeshVertex> vertexBuffer;
			Matrix4 pose = Math::Identity();
			Size2 tileCount = { 0u, 0u };
			Reference<TerrainRenderer> terrain;
		};

		struct VirtualViewport : public virtual ViewportDescriptor {
			Matrix4 view = Math::Identity();
			Matrix4 projection = Math::Orthographic(10.0f, 1.0f, -5.0f, 5.0f);

			inline VirtualViewport(SceneContext* context) : ViewportDescriptor(context) {}
			inline virtual ~VirtualViewport() {}
			inline virtual Matrix4 ViewMatrix()const override { return view; }
			inline virtual Matrix4 ProjectionMatrix()const override { return projection; }
			inline virtual Vector4 ClearColor()const { return Vector4(1.0f); }
		};

		struct LodRange {
			Vector2 lodStart = {};
			Size2 lodTileCount = {};
		};

		struct PerInstanceData {
			alignas(16) Matrix4 pose = Math::Identity();

		};


		class TerrainViewportData : public virtual GraphicsObjectDescriptor::ViewportData {
		private:
			const Reference<const GraphicsObjectData> m_data;
			const Reference<const RendererFrustrumDescriptor> m_frustrum;
			Graphics::ArrayBufferReference<PerInstanceData> m_instanceBuffer;
			size_t m_instanceCount = 0u;

			inline uint32_t RoundDetailLodTileCount(const uint32_t detailLodTileCount) {
				return Math::Max((detailLodTileCount + 3u) / 4u, 1u) * 4u;
			}

			inline uint32_t CellCountForDetailLod(const uint32_t detailLodTileCount) {
				uint32_t cnt = RoundDetailLodTileCount(detailLodTileCount);
				return cnt * cnt;
			}

			inline LodRange ComputeLodRange(const Vector3& localViewPos, const float tileSize, const float lodTileSize, const uint32_t detailLodTileCount) {
				const Vector2 viewCellPos = (Vector2(localViewPos.x, localViewPos.z) + 0.5f * tileSize) / lodTileSize;
				const Vector2 viewPosCell = Vector2(std::floorf(viewCellPos.x), std::floorf(viewCellPos.y));
				const Vector2 higherLodCornerCellPos = (viewPosCell + 1.0f) * 0.5f;
				const Vector2 higherLodCornerCell = Vector2(std::floorf(higherLodCornerCellPos.x), std::floorf(higherLodCornerCellPos.y));

				const float halfTileCount = static_cast<float>(detailLodTileCount) * 0.5f;
				const Vector2 higherLodStartCell = higherLodCornerCell - (halfTileCount * 0.5f);
				const Vector2 higerLodEndCell = higherLodCornerCell + (halfTileCount * 0.5f);

				LodRange range = {};
				range.lodStart = Vector2(std::floor(higherLodStartCell.x), std::floor(higherLodStartCell.y)) * 2.0f;
				range.lodTileCount = Size2(Vector2(std::ceil(higerLodEndCell.x), std::ceil(higerLodEndCell.y)) * 2.0f - range.lodStart);
				assert(range.lodTileCount.x <= RoundDetailLodTileCount(detailLodTileCount));
				assert(range.lodTileCount.y <= RoundDetailLodTileCount(detailLodTileCount));
				return range;
			}

			inline PerInstanceData* FillLod(
				const Vector3& localViewPos, const Matrix4& pose,
				const Vector2& gridOffset, const float tileSize,
				const uint32_t lodId, const uint32_t lodCount, const uint32_t detailLodTileCount,
				PerInstanceData* lodBase) {
				
				// Lod metrics:
				const float lodSize = std::powf(0.5f, float(lodId));
				const float lodTileSize = (lodSize * tileSize);
				const Vector2 lodCellCount = Vector2(
					std::round(m_data->tileCount.x / lodSize),
					std::round(m_data->tileCount.y / lodSize));
				const Vector2 lodGridOffset = ((lodCellCount - 1.0f) * -0.5f) * lodTileSize - gridOffset;

				// 'Area' to fill:
				LodRange lodRange;
				if (lodId <= 0u) {
					lodRange.lodStart = Size2(0.0f, 0.0f);
					lodRange.lodTileCount = m_data->tileCount;
				}
				else lodRange = ComputeLodRange(localViewPos, tileSize, lodSize * tileSize, detailLodTileCount);

				// 'Area' filled by the more detailed LOD:
				Vector2 detailLodStart = Vector2(-8.0f);
				Vector2 detailLodEnd = detailLodStart;
				if (lodId < lodCount) {
					const LodRange detailLodRange = ComputeLodRange(localViewPos, tileSize, lodTileSize * 0.5f, detailLodTileCount);
					detailLodStart = detailLodRange.lodStart * 0.5f;
					detailLodEnd = detailLodStart + Vector2(detailLodRange.lodTileCount) * 0.5f;
					assert((detailLodStart.x + std::numeric_limits<float>::epsilon()) >= lodRange.lodStart.x);
					assert((detailLodStart.y + std::numeric_limits<float>::epsilon()) >= lodRange.lodStart.y);
					assert(detailLodEnd.x <= (lodRange.lodStart.x + lodRange.lodTileCount.x + std::numeric_limits<float>::epsilon()));
					assert(detailLodEnd.y <= (lodRange.lodStart.y + lodRange.lodTileCount.y + std::numeric_limits<float>::epsilon()));
				}

				// Place each cell:
				// Note: This loop can easily run within a compute kernel; 
				// likely optimal strategy would be to make a potentially combined kernel for the lowest density LOD-s of all terrains,
				// while keeping the finite amount of high-density LOD tile computations on host 
				// to prevent unnecessary kernel load when it comes to task binary search (for combined) or kernel launch overhead (for individual dispatch case).
				for (uint32_t y = 0u; y < lodRange.lodTileCount.y; y++)
					for (uint32_t x = 0u; x < lodRange.lodTileCount.x; x++) {
						// Cell position:
						const Vector2 cellId = lodRange.lodStart + Vector2(float(x), float(y));
						const Vector2 tilePos = cellId * lodTileSize + lodGridOffset;

						// 'Dead' if out of bounds or filled-in by more dense LOD:
						const bool dead = (
							cellId.x < 0.0f || cellId.y < 0.0f ||
							cellId.x >= lodCellCount.x || cellId.y >= lodCellCount.y || (
								cellId.x >= detailLodStart.x && cellId.y >= detailLodStart.y &&
								cellId.x < detailLodEnd.x && cellId.y < detailLodEnd.y));

						// __TODO__: Add frustrum culling..

						// For now, we don't have a indirect draw buffer, so scale will do:
						const float scl = dead ? 0.0f : lodSize;

						// Fill in data:
						PerInstanceData& data = lodBase[lodRange.lodTileCount.x * y + x];
						data.pose = pose * Matrix4(
							Vector4(scl, 0.0f, 0.0f, 0.0f),
							Vector4(0.0f, scl, 0.0f, 0.0f),
							Vector4(0.0f, 0.0f, scl, 0.0f),
							Vector4(tilePos.x, 0.0, tilePos.y, 1.0f));
					}

				// Next LOD will start after the current tiles, directly:
				return lodBase + size_t(lodRange.lodTileCount.x * lodRange.lodTileCount.y);
			}

		public:
			inline TerrainViewportData(const GraphicsObjectData* data, const RendererFrustrumDescriptor* frustrum)
				: GraphicsObjectDescriptor::ViewportData(Graphics::GraphicsPipeline::IndexType::TRIANGLE)
				, m_data(data), m_frustrum(frustrum) {
				assert(m_data != nullptr);
				assert(m_frustrum != nullptr);
			}

			inline virtual ~TerrainViewportData() {}
			
			inline static Reference<TerrainViewportData> Create(const GraphicsObjectData* data, const RendererFrustrumDescriptor* frustrum) {
				return Object::Instantiate<TerrainViewportData>(data, frustrum);
			}


			// Temporary... Once this becomes per-viewport, we will have this one added as a separate job or event.
			inline void Update() {
				// If terrain is dead, we just makes ure nothing gets rendered:
				if (m_data->terrain == nullptr || m_data->indexBuffer == nullptr) {
					m_instanceBuffer = nullptr;
					return;
				}

				// Failure callback:
				auto fail = [&](const auto&... message) -> void {
					m_data->terrain->Context()->Log()->Error("TerrainRenderer::Helpers::TerrainViewportData::Update - ", message...);
					m_instanceBuffer = nullptr;
				};

				// Grid pose and origin:
				const float tileSize = static_cast<float>(m_data->indexBuffer->TileGridSize());
				const Vector2 gridOffset = ((Vector2(m_data->tileCount) - 1.0f) * -0.5f) * tileSize;
				const Matrix4 pose = m_data->pose * Matrix4(
					Vector4(1.0f, 0.0f, 0.0f, 0.0f),
					Vector4(0.0f, 1.0f, 0.0f, 0.0f),
					Vector4(0.0f, 0.0f, 1.0f, 0.0f),
					Vector4(gridOffset.x, 0.0f, gridOffset.y, 1.0f));

				// View data:
				const Vector3 viewPos = m_frustrum->EyePosition();
				const Vector3 localViewPos = Math::Inverse(pose) * Vector4(viewPos, 1.0f);

				// LOD-s:
				const uint32_t lodCount = m_data->terrain->DetailLodCount();
				const uint32_t detailLodTileCount = m_data->terrain->DetailLodTileCount();

				// Make sure we have large enough of an instance buffer:
				const size_t baseInstanceCount = m_data->tileCount.x * m_data->tileCount.y;
				const size_t maxPerLodInstanceCount = CellCountForDetailLod(detailLodTileCount);
				const size_t lodInstanceCount = lodCount * maxPerLodInstanceCount;
				const size_t totalInstanceCount = baseInstanceCount + lodInstanceCount;
				if (m_instanceBuffer == nullptr || (m_instanceBuffer->Size() / sizeof(PerInstanceData)) != totalInstanceCount) {
					m_instanceBuffer = m_data->terrain->Context()->Graphics()->Device()->CreateArrayBuffer<PerInstanceData>(totalInstanceCount);
					if (m_instanceBuffer == nullptr)
						return fail("Failed to create instance buffer! [File: ", __FILE__, "; Line: ", __LINE__, "]");
				}

				// Down the line, we will need some more optimized in-flight buffers, but for now, this shall suffisce:
				{
					PerInstanceData* const instanceData = m_instanceBuffer.Map();
					if (instanceData == nullptr)
						return fail("Failed to map instance buffer! [File: ", __FILE__, "; Line: ", __LINE__, "]");

					PerInstanceData* lodBase = instanceData;
					for (uint32_t detail = 0u; detail <= lodCount; detail++) {
						lodBase = FillLod(localViewPos, pose, gridOffset, tileSize, detail, lodCount, detailLodTileCount, lodBase);
						assert(size_t(lodBase - instanceData) <= totalInstanceCount);
					}
					m_instanceCount = (lodBase - instanceData);

					m_instanceBuffer->Unmap(true);
				}
			}

			inline virtual Graphics::BindingSet::BindingSearchFunctions BindingSearchFunctions()const override {
				return m_data->cachedMaterialInstance->BindingSearchFunctions();
			}

			inline virtual void GetGeometry(GraphicsObjectDescriptor::GeometryDescriptor& descriptor)const override {
				// Vertex fields:
				{
					const auto& vertexBuffer = m_data->vertexBuffer;
					const uint32_t vertexCount = (vertexBuffer == nullptr) ? 0u : static_cast<uint32_t>(vertexBuffer->Size() / sizeof(MeshVertex));
					auto setVertexField = [&](GraphicsObjectDescriptor::PerVertexBufferData& data, size_t offset) {
						data.buffer = vertexBuffer;
						data.bufferOffset = static_cast<uint32_t>(offset);
						data.numEntriesPerInstance = vertexCount;
						data.perVertexStride = static_cast<uint32_t>(sizeof(MeshVertex));
						data.perInstanceStride = 0u;
					};
					setVertexField(descriptor.vertexPositions, offsetof(MeshVertex, position));
					setVertexField(descriptor.vertexNormals, offsetof(MeshVertex, normal));
					setVertexField(descriptor.vertexUVs, offsetof(MeshVertex, uv));
				}


				// Index buffer:
				{
					descriptor.indexBuffer.buffer = (m_data->indexBuffer == nullptr) ? nullptr : m_data->indexBuffer->Buffer();
					descriptor.indexBuffer.baseIndexOffset = 0u;
					descriptor.indexBuffer.indexCount = (descriptor.indexBuffer.buffer == nullptr) ? 0u :
						static_cast<uint32_t>(descriptor.indexBuffer.buffer->Size() / sizeof(uint32_t));
				}

				// JM_ObjectTransform:
				{
					descriptor.instanceTransforms.buffer = m_instanceBuffer;
					descriptor.instanceTransforms.bufferOffset = static_cast<uint32_t>(offsetof(PerInstanceData, pose));
					descriptor.instanceTransforms.elemStride = static_cast<uint32_t>(sizeof(PerInstanceData));
				}

				// __TODO__: Shader will likely need the uv tiling and offset.

				// Instances:
				{
					// __TODO__: Once we add culling, we will need to add the indirect draw command(s) as usual.
					descriptor.instances.count = (m_instanceBuffer == nullptr) ? 0u : static_cast<uint32_t>(m_instanceCount);
					descriptor.instances.liveInstanceEntryCount = descriptor.instances.count;
				}

				// Flags:
				{
					descriptor.flags = GraphicsObjectDescriptor::GeometryFlags::VERTEX_POSITION_CONSTANT;
				}
			}

			inline virtual Reference<Component> GetComponent(size_t)const override {
				return m_data->terrain;
			}
		};


		class GraphicsObject 
			: public virtual GraphicsObjectDescriptor
			, public virtual JobSystem::Job {
		private:
			const Reference<GraphicsObjectData> m_data;
			const Reference<VirtualViewport> m_frustrum;
			Reference<TerrainViewportData> m_viewData;

			inline GraphicsObject(GraphicsObjectData* data)
				: GraphicsObjectDescriptor(data->cachedMaterialInstance->Shader(), data->terrain->Layer())
				, m_data(data), m_frustrum(Object::Instantiate<VirtualViewport>(data->terrain->Context())) {
				m_viewData = TerrainViewportData::Create(data, m_frustrum);
			}

		public:
			inline virtual ~GraphicsObject() {}

			inline static Reference<GraphicsObject> Create(TerrainRenderer* terrain) {
				assert(terrain != nullptr);
				// Failure callback:
				const auto fail = [&](const auto&... message) {
					terrain->Context()->Log()->Error("TerrainRenderer::Helpers::GraphicsObject::Create - ", message...);
					return nullptr;
				};
				
				// Obtain material:
				Reference<const Material::Instance> material = terrain->MaterialInstance();
				if (material == nullptr)
					material = SampleDiffuseShader::MaterialInstance(terrain->Context());
				if (material == nullptr)
					return fail("Failed to obtain material instance! [File: ", __FILE__, "; Line: ", __LINE__, "]");

				// Create common resources:
				const Reference<GraphicsObjectData> data = Object::Instantiate<GraphicsObjectData>();
				assert(data != nullptr);
				data->cachedMaterialInstance = material->CreateCachedInstance();
				if (data->cachedMaterialInstance == nullptr)
					return fail("Failed to create cached material instance! [File: ", __FILE__, "; Line: ", __LINE__, "]");
				data->terrain = terrain;

				// Done:
				Reference<GraphicsObject> object = new GraphicsObject(data);
				assert(object != nullptr);
				object->ReleaseRef();
				assert(object->RefCount() == 1u);
				return object;
			}

			virtual Reference<const ViewportData> GetViewportData(const RendererFrustrumDescriptor* frustrum) {
				Unused(frustrum); // For now, we only debug with a dedicated transform, so this is a temporary thing.
				return m_viewData;
			}


		protected:
			virtual void Execute()override {
				// Make sure terrain is alive:
				if (m_data->terrain == nullptr || m_data->terrain->Destroyed()) {
					m_data->terrain = nullptr;
					m_data->indexBuffer = nullptr;
					m_data->vertexBuffer = nullptr;
					return;
				}

				const auto fail = [&](const auto&... message) {
					m_data->terrain->Context()->Log()->Error("TerrainRenderer::Helpers::GraphicsObject::Create - ", message...);
					m_data->indexBuffer = nullptr;
					m_data->vertexBuffer = nullptr;
				};
				
				// Refresh material:
				m_data->cachedMaterialInstance->Update();

				// Refresh index and vertex buffers if needed:
				const uint32_t tileDensity = m_data->terrain->TileDensity();
				m_data->tileCount = Size2(m_data->terrain->m_tileCount);
				if (m_data->indexBuffer == nullptr || m_data->indexBuffer->TileGridSize() != tileDensity) {
					m_data->indexBuffer = TileIndexBuffer::Get(
						m_data->terrain->Context()->Log(), 
						m_data->terrain->Context()->Graphics()->Device(), 
						tileDensity);
					if (m_data->indexBuffer == nullptr)
						fail("Failed to get tile index buffer! [File: ", __FILE__, "; Line: ", __LINE__, "]");
					else {
						// __TODO__: Once we have the tiles 'arranging themselves' to the viewport as intended, 
						// we can safetly move from here to actual, real tiles that take-up shape of the heightmap.
						m_data->vertexBuffer = CreateVertexBuffer(
							m_data->terrain->Context()->Log(), 
							m_data->terrain->Context()->Graphics()->Device(), 
							m_data->indexBuffer->TileGridSize());
						if (m_data->vertexBuffer == nullptr)
							fail("Failed to get tile vertex buffer! [File: ", __FILE__, "; Line: ", __LINE__, "]");
					}
				}

				// Refresh pose:
				{
					const Transform* const transform = m_data->terrain->GetTransform();
					m_data->pose = (transform == nullptr) ? Math::Identity() : transform->FrameCachedWorldMatrix();
				}

				// Update the view-data (temporary; Once we have true per-viewport information, we will change this to something meaningful):
				{
					Reference<Transform> view = m_data->terrain->m_debugObserver;
					m_frustrum->view = (view != nullptr)
						? Math::Inverse(view->FrameCachedWorldMatrix())
						: m_data->pose;
					m_viewData->Update();
				}
			}

			virtual void CollectDependencies(Callback<Job*> addDependency) override {
				Unused(addDependency);
			}
		};
#pragma endregion





#pragma region Enable/Disable
		inline static void DisableRenderer(TerrainRenderer* self) {
			assert(self != nullptr);

			// Remove synch job:
			const Reference<GraphicsObjectDescriptor::Set::ItemOwner> objectRef = self->m_graphicsObject;
			if (objectRef != nullptr) {
				const Reference<GraphicsObject> object = dynamic_cast<GraphicsObject*>(objectRef->Item());
				assert(object != nullptr);
				self->Context()->Graphics()->SynchPointJobs().Remove(object);
			}
			else assert(self->m_graphicsObject == nullptr);

			// Remove from object set:
			const Reference<GraphicsObjectDescriptor::Set> objectSet = self->m_graphicsObjectSet;
			if (objectSet == nullptr)
				assert(self->m_graphicsObjectSet == nullptr);
			if (objectSet != nullptr && objectRef != nullptr)
				objectSet->Remove(objectRef);

			// Remove dangling references:
			self->m_graphicsObject = nullptr;
			self->m_graphicsObjectSet = nullptr;
		}

		inline static void CreateRendererIfNotPresent(TerrainRenderer* self) {
			assert(self != nullptr);
			assert(self->ActiveInHierarchy());

			// If everything's in order, we skip:
			if (self->m_graphicsObject != nullptr && self->m_graphicsObjectSet != nullptr)
				return;

			// Cleanup, just in case there's an internal error:
			DisableRenderer(self);
			assert(self->m_graphicsObject == nullptr);

			// Obtain graphics object set:
			Reference<GraphicsObjectDescriptor::Set> objectSet = self->m_graphicsObjectSet;
			if (objectSet == nullptr) {
				assert(self->m_graphicsObjectSet == nullptr);
				objectSet = GraphicsObjectDescriptor::Set::GetInstance(self->Context());
			}

			// Create a new renderer and add it to systems:
			const Reference<GraphicsObject> object = GraphicsObject::Create(self);
			if (object == nullptr)
				return;
			const Reference<GraphicsObjectDescriptor::Set::ItemOwner> ref = Object::Instantiate<GraphicsObjectDescriptor::Set::ItemOwner>(object);
			self->Context()->Graphics()->SynchPointJobs().Add(object);
			self->m_graphicsObject = ref;
			objectSet->Add(ref);
			self->m_graphicsObjectSet = objectSet;
		}

		inline static void ReinitializeRenderer(TerrainRenderer* self) {
			assert(self != nullptr);
			assert(self->ActiveInHierarchy());

			// Make sure we don't enter here with an existing renderer:
			DisableRenderer(self);
			assert(self->m_graphicsObject == nullptr);
			assert(self->m_graphicsObjectSet == nullptr);

			// Create a new one:
			CreateRendererIfNotPresent(self);
		}

		inline static void OnObjectDirty(TerrainRenderer* self) {
			assert(self != nullptr);
			if (self->ActiveInHierarchy())
				CreateRendererIfNotPresent(self);
			else DisableRenderer(self);
		}

		inline static void ScheduleGraphicsObjectRefresh(TerrainRenderer* self) {
			assert(self != nullptr);
			std::unique_lock<std::recursive_mutex> lock(self->Context()->UpdateLock());
			if (self->m_dirty)
				return;
			void(*invokeOnTriMeshRendererDirty)(Object*) = [](Object* selfPtr) {
				TerrainRenderer* self = dynamic_cast<TerrainRenderer*>(selfPtr);
				self->m_dirty = false;
				if (self->ActiveInHierarchy())
					ReinitializeRenderer(self);
				else DisableRenderer(self);
			};
			self->Context()->ExecuteAfterUpdate(Callback<Object*>(invokeOnTriMeshRendererDirty), self);
			self->m_dirty = true;
		}

		inline static void RecreateOnMaterialInstanceInvalidated(TerrainRenderer* self, const Jimara::Material* material) {
			std::unique_lock<std::recursive_mutex> lock(self->Context()->UpdateLock());
			if (material != nullptr) {
				if (material == self->m_material) {
					Material::Reader reader(material);
					self->m_materialInstance = reader.SharedInstance();
				}
			}
			if (self->ActiveInHierarchy())
				ReinitializeRenderer(self);
			else DisableRenderer(self);
		}
#pragma endregion
	};





#pragma region Component
	TerrainRenderer::TerrainRenderer(Component* parent, const std::string_view& name) : Component(parent, name) {}

	TerrainRenderer::~TerrainRenderer() {
		Helpers::DisableRenderer(this);
	}

	void TerrainRenderer::SetMaterial(Jimara::Material* material) {
		std::unique_lock<std::recursive_mutex> lock(Context()->UpdateLock());
		if (Destroyed()) 
			material = nullptr;
		if (material == m_material) 
			return;
		if (m_material != nullptr)
			m_material->OnInvalidateSharedInstance() -= Callback(&Helpers::RecreateOnMaterialInstanceInvalidated, this);
		m_material = material;
		if (m_material != nullptr) {
			if (!Destroyed())
				m_material->OnInvalidateSharedInstance() += Callback(&Helpers::RecreateOnMaterialInstanceInvalidated, this);
			Jimara::Material::Reader reader(material);
			Reference<const Jimara::Material::Instance> instance = reader.SharedInstance();
			if (instance == m_materialInstance) 
				return; // Stuff will auto-resolve in this case
			m_materialInstance = instance;
		}
		else m_materialInstance = nullptr;
		Helpers::ScheduleGraphicsObjectRefresh(this);
	}

	const Jimara::Material::Instance* TerrainRenderer::MaterialInstance() {
		if (m_materialInstance == nullptr)
			m_materialInstance = SampleDiffuseShader::MaterialInstance(Context());
		return m_materialInstance;
	}

	void TerrainRenderer::SetMaterialInstance(const Jimara::Material::Instance* materialInstance) {
		std::unique_lock<std::recursive_mutex> lock(Context()->UpdateLock());
		if (Destroyed()) materialInstance = nullptr;
		if (m_material != nullptr) SetMaterial(nullptr);
		else if (m_materialInstance == materialInstance) return;
		m_materialInstance = materialInstance;
		Helpers::ScheduleGraphicsObjectRefresh(this);
	}

	void TerrainRenderer::SetLayer(Jimara::Layer layer) {
		if (m_layer == layer)
			return;
		m_layer = layer;
		Helpers::ScheduleGraphicsObjectRefresh(this);
	}

	void TerrainRenderer::SetTileDensity(uint32_t density) {
		if (density < 1u)
			density = 1u;
		if (m_tileDensity == density)
			return;
		const auto prevResolution = TerrainResolution();
		m_tileDensity = density;
		SetTerrainResolution(prevResolution);
	}

	void TerrainRenderer::SetLodDetailLodCount(uint32_t count) {
		if (m_detailLodCount == count)
			return;
		m_detailLodCount = count;
	}

	void TerrainRenderer::SetDetailLodTileCount(uint32_t count) {
		if (m_detailLodTileCount == count)
			return;
		m_detailLodTileCount = count;
	}

	void TerrainRenderer::SetTerrainResolution(uint32_t resolution) {
		const auto tileCount = ((resolution + m_tileDensity - 1u) / m_tileDensity);
		if (tileCount == m_tileCount)
			return;
		m_tileCount = tileCount;
	}

	void TerrainRenderer::GetFields(Callback<Serialization::SerializedObject> recordElement) {
		Component::GetFields(recordElement);
		JIMARA_SERIALIZE_FIELDS(this, recordElement) {
			JIMARA_SERIALIZE_FIELD_GET_SET(Material, SetMaterial, "Material", "Material to render with.");
			JIMARA_SERIALIZE_FIELD_GET_SET(Layer, SetLayer, "Layer", "Graphics object layer (for renderer filtering)", Layers::LayerAttribute::Instance());
			JIMARA_SERIALIZE_FIELD_GET_SET(TileDensity, SetTileDensity, "Tile Density",
				"Number of square cells aross a single terrain patch. \n"
				"Terrain will be rendered as a grid of tiles of various LOD-s; Each will be a square of TileDensity() x TileDensity();");
			JIMARA_SERIALIZE_FIELD_GET_SET(DetailLodCount, SetLodDetailLodCount, "Detail LOD count", "Number of levels of detail.");
			JIMARA_SERIALIZE_FIELD_GET_SET(DetailLodTileCount, SetDetailLodTileCount, "Detail LOD Tile Count", 
				"Minimal grid size for each detail LOD grid of grids.");
			JIMARA_SERIALIZE_FIELD_GET_SET(TerrainResolution, SetTerrainResolution, "Terrain Resolution",
				"Total terrain resolution in cells. \n"
				"Both x and y have to be multiples of TileDensity(). \n"
				"When changine either parameter, the resolution will be rounded up towards the smallest multiple of TileDensity().");
			JIMARA_SERIALIZE_FIELD(m_debugObserver, "Debug Observer", "[Temporary] Transform, representing a viewport position; for LOD visualization.");
		};
	}

	void TerrainRenderer::GetSerializedActions(Callback<Serialization::SerializedCallback> report) {
		Component::GetSerializedActions(report);
		// __TODO__: Implement this crap!
	}

	AABB TerrainRenderer::GetBoundaries()const {
		const Transform* transform = GetTransform();
		if (transform == nullptr)
			return {};
		const Vector3 pos = transform->WorldPosition();
		return AABB(pos, pos);
	}

	void TerrainRenderer::OnComponentInitialized() {
		Helpers::ScheduleGraphicsObjectRefresh(this);
	}

	void TerrainRenderer::OnComponentEnabled() {
		Helpers::ScheduleGraphicsObjectRefresh(this);
	}

	void TerrainRenderer::OnComponentDisabled() {
		Helpers::OnObjectDirty(this);
	}

	void TerrainRenderer::OnComponentDestroyed() {
		Helpers::DisableRenderer(this);
	}

	template<> JIMARA_API void TypeIdDetails::GetTypeAttributesOf<TerrainRenderer>(const Callback<const Object*>& report) {
#if false
		static const Reference<ComponentFactory> factory = ComponentFactory::Create<TerrainRenderer>(
			"Terrain Renderer", "Jimara/Graphics/TerrainRenderer", "Terrain Renderer component");
		report(factory);
#endif
	}
#pragma endregion
}
