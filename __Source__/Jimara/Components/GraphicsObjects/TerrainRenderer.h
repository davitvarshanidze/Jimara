#pragma once
#include "../Transform.h"
#include "../../Data/Materials/Material.h"
#include "../../Environment/Layers.h"
#include "../../Environment/Interfaces/BoundedObject.h"


namespace Jimara {
	/// <summary> Let the engine know this class exists </summary>
	JIMARA_REGISTER_TYPE(Jimara::TerrainRenderer);

	/// <summary>
	/// A renderer, displying a terrain in the scene.
	/// </summary>
	class JIMARA_API TerrainRenderer : public virtual Component, virtual BoundedObject {
	public:
		/// <summary>
		/// Constructor
		/// </summary>
		/// <param name="parent"> Parent component </param>
		/// <param name="name"> Component name </param>
		TerrainRenderer(Component* parent, const std::string_view& name = "TerrainRenderer");

		/// <summary> Virtual destructor </summary>
		virtual ~TerrainRenderer();

		/// <summary> Material to render with </summary>
		inline Jimara::Material* Material()const { return m_material; }

		/// <summary>
		/// Sets new material to use
		/// </summary>
		/// <param name="material"> New material </param>
		void SetMaterial(Jimara::Material* material);

		/// <summary> Material instance the renderer uses </summary>
		const Jimara::Material::Instance* MaterialInstance();

		/// <summary>
		/// Sets new material instance to use (will discard the Material connection)
		/// </summary>
		/// <param name="materialInstance"> New material instance </param>
		void SetMaterialInstance(const Jimara::Material::Instance* materialInstance);

		/// <summary> Graphics object layer (for renderer filtering) </summary>
		inline Jimara::Layer Layer()const { return m_layer; }

		/// <summary>
		/// Sets graphics object layer (for renderer filtering)
		/// </summary>
		/// <param name="layer"> New layer </param>
		void SetLayer(Jimara::Layer layer);

		/// <summary>
		/// Number of square cells aross a single terrain patch.
		/// <para/> Terrain will be rendered as a grid of tiles of various LOD-s; Each will be a square of TileDensity() x TileDensity();
		/// </summary>
		uint32_t TileDensity()const { return m_tileDensity; }

		/// <summary>
		/// Sets tile density.
		/// </summary>
		/// <param name="density"> New value (will be clamped to 1 from below) </param>
		void SetTileDensity(uint32_t density);

		/// <summary> Number of levels of detail. </summary>
		uint32_t DetailLodCount()const { return m_detailLodCount; }

		/// <summary>
		/// Sets the number of levels of detail.
		/// </summary>
		/// <param name="count"> LOD level count </param>
		void SetLodDetailLodCount(uint32_t count);

		/// <summary> Minimal grid size for each detail LOD grid of grids. </summary>
		inline uint32_t DetailLodTileCount()const { return m_detailLodTileCount; }

		/// <summary>
		/// Sets minimal grid size for each detail LOD grid of grids.
		/// </summary>
		/// <param name="count"> Minimal grid size for each detail LOD grid of grids to use. </param>
		void SetDetailLodTileCount(uint32_t count);

		/// <summary>
		/// Total terrain resolution in cells.
		/// <para/> Has to be multiple of TileDensity(). 
		/// When changine either parameter, the resolution will be rounded up towards the smallest multiple of TileDensity().
		/// </summary>
		uint32_t TerrainResolution()const { return m_tileCount * m_tileDensity; }

		/// <summary>
		/// Sets terrain resolution.
		/// <para/> Has to be multiple of TileDensity(). 
		/// When changine either parameter, the resolution will be rounded up towards the smallest multiple of TileDensity().
		/// </summary>
		/// <param name="resolution"> Terrain dimentions to use. </param>
		void SetTerrainResolution(uint32_t resolution);

		/// <summary>
		/// Exposes fields to serialization utilities
		/// </summary>
		/// <param name="recordElement"> Reports elements with this </param>
		virtual void GetFields(Callback<Serialization::SerializedObject> recordElement)override;

		/// <summary>
		/// Reports actions associated with the component.
		/// </summary>
		/// <param name="report"> Actions will be reported through this callback </param>
		virtual void GetSerializedActions(Callback<Serialization::SerializedCallback> report)override;

		/// <summary> Retrieves object boundaries </summary>
		virtual AABB GetBoundaries()const override;

	protected:
		/// <summary> Invoked by the scene on the first frame this component gets instantiated </summary>
		virtual void OnComponentInitialized()override;

		/// <summary> Invoked, whenever the component becomes active in herarchy </summary>
		virtual void OnComponentEnabled()override;

		/// <summary> Invoked, whenever the component stops being active in herarchy </summary>
		virtual void OnComponentDisabled()override;

		/// <summary> We need to ghet rid of the graphics object, tied to the scene, once the component is destroyed </summary>
		virtual void OnComponentDestroyed()override;

	private:
		// Underlying graphics object.
		Reference<Object> m_graphicsObjectSet;
		Reference<Object> m_graphicsObject;

		// Material to render with
		Reference<Jimara::Material> m_material;

		// Targetted material instance
		Reference<const Jimara::Material::Instance> m_materialInstance;

		// Layer.
		Jimara::Layer m_layer = 0u;

		// Tile settings.
		uint32_t m_tileDensity = 8u;
		uint32_t m_tileCount = 16u;
		uint32_t m_detailLodCount = 2u;
		uint32_t m_detailLodTileCount = 4u;

		// Temporary debug observer transform.
		WeakReference<Transform> m_debugObserver;

		// True, if graphics object recreation call is scheduled.
		std::atomic_bool m_dirty = false;

		// Underlying implementation goes here.
		struct Helpers;
	};

	// Type detail callbacks
	template<> inline void TypeIdDetails::GetParentTypesOf<TerrainRenderer>(const Callback<TypeId>& report) {
		report(TypeId::Of<Component>());
		report(TypeId::Of<BoundedObject>());
	}
	template<> JIMARA_API void TypeIdDetails::GetTypeAttributesOf<TerrainRenderer>(const Callback<const Object*>& report);
}