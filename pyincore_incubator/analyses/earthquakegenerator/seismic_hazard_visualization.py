# Copyright (c) 2024 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/
import numpy as np
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_origin


class SeismicHazardVisualization:
    def __init__(self, setup, scenarios_dict, num_scenarios, return_period):
        """
        Initializes the SeismicHazardVisualization with necessary parameters for visualization.

        Parameters:
        -----------
        setup : object
            An instance of setup class containing site and source data.
        scenarios_dict : dict
            A dictionary of intensity measures for each site and scenario.
        return_period : int
            The return period for seismic hazard analysis in years.
        """
        self.setup = setup
        self.scenarios_dict = scenarios_dict
        self.num_scenarios = num_scenarios
        self.return_period = return_period

    def prepare_im_data(self, exceedance_probability, ground_motion_type):
        """
        Prepare the data for plotting based on the specified exceedance probability and ground motion type.

        Parameters:
        -----------
        exceedance_probability : float
            The exceedance probability for which to prepare the data.
        ground_motion_type : str
            The type of ground motion ('PGA' or 'PGV') to prepare data for.

        Returns:
        --------
        list
            A list of tuples containing longitude, latitude, and exceedance value for each site.
        """
        annual_exceedance_probability = 1 - (1 - exceedance_probability) ** (
            1 / self.return_period
        )

        processed_data = []
        for site_id in self.setup.site_data["guid"]:
            # Filter intensity measures for the current site and ground motion type
            site_im_values = [
                im[ground_motion_type]
                for (idx, id, *rest), im in self.scenarios_dict.items()
                if id == site_id and ground_motion_type in im
            ]

            exceedance_value = np.percentile(
                site_im_values, 100 * (1 - annual_exceedance_probability)
            )

            try:
                row = self.setup.site_data[
                    self.setup.site_data["guid"] == site_id
                ].iloc[0]
                lon, lat = row["geometry"].x, row["geometry"].y
                processed_data.append((lon, lat, exceedance_value))
            except IndexError:
                print(f"Site ID {site_id} not found in site_data")

        return processed_data

    def generate_continuous_earthquake_geotiff(
        self,
        exceedance_probability,
        ground_motion_type,
        grid_resolution,
        geotiff_file_name,
    ):
        """
        Generates a continuous raster GeoTIFF representing earthquake exceedance values over an area,
        based on interpolation of exceedance data at specific sites.

        Parameters:
        -----------
        exceedance_probability : float
            The probability threshold for selecting sites, expressed as a decimal (e.g., 0.1 for 10% chance).
        ground_motion_type : str
            The type of ground motion ('PGA' or 'PGV') for which data is prepared.
        grid_resolution : float
            The resolution of the grid in degrees, determining the spacing of points in the mesh.
        geotiff_file_name : str
            The filename of the GeoTIFF.

        Returns:
        --------
        None
            Creates a GeoTIFF file with the interpolated raster data.
        """

        # Prepare intensity measure data
        data = self.prepare_im_data(exceedance_probability, ground_motion_type)
        lons, lats, ims = zip(*data)

        # Create a mesh grid covering the entire area of interest
        xi = np.arange(min(lons), max(lons), grid_resolution)
        yi = np.arange(min(lats), max(lats), grid_resolution)
        X, Y = np.meshgrid(xi, yi)

        # Interpolate intensity measures onto the mesh grid
        Z = griddata((lons, lats), ims, (X, Y), method="cubic")

        # Define the transformation for the GeoTIFF
        # west, north = X.min(), Y.max()
        west, north = X.min(), Y.min()
        pixel_width, pixel_height = grid_resolution, -grid_resolution
        transform = from_origin(west, north, pixel_width, pixel_height)

        # Write the interpolated grid to a GeoTIFF file
        # file_name = f"Continuous_{ground_motion_type}.tif"
        # file_path = f"{geotiff_file_path}/{file_name}"
        with rasterio.open(
            geotiff_file_name,
            "w",
            driver="GTiff",
            height=Z.shape[0],
            width=Z.shape[1],
            count=1,
            dtype=str(Z.dtype),
            crs="+proj=latlong",
            transform=transform,
        ) as dst:
            dst.write(Z, 1)

        # print(f"GeoTIFF file created at: {file_path}")
        print(f"GeoTIFF file created at: {geotiff_file_name}")


# i def plot_distances_contour_map(self, distance_type):
#         """
#         Plot a contour map representing distances (rupture, Joyner-Boore, horizontal, Haversine).
#
#         Parameters:
#         -----------
#         distance_type : str
#             The type of distance to plot ('rupture', 'joyner_boore', 'horizontal', 'haversine').
#         sources : list
#             The list of sources to use for distance calculations.
#         """
#         _ , sources = self.earthquake_generator.generate_scenarios(num_scenarios=self.num_scenarios)
#
#         # Calculate distances for each source
#         for i, source in enumerate(sources):
#             distance_data = []
#
#             source_info = self.setup.source_data[source]
#             source_loc = (source_info['lon'], source_info['lat'], source_info['depth'])
#
#             for j, site_id in enumerate(self.setup.site_id):
#                 site_loc = (self.setup.site_lon[j], self.setup.site_lat[j], self.setup.site_depth[j])
#
#                 # Calculate distances based on the distance type
#                 if distance_type == 'rupture':
#                     distance = SeismicDistanceCalculator.calculate_rupture_distance(site_loc, source_loc, source_info['strike'], source_info['dip'])
#                 elif distance_type == 'joyner_boore':
#                     distance = SeismicDistanceCalculator.calculate_joyner_boore_distance(site_loc, source_loc)
#                 elif distance_type == 'horizontal':
#                     distance = SeismicDistanceCalculator.calculate_horizontal_distance(site_loc, source_loc, source_info['strike'])
#                 elif distance_type == 'haversine':
#                     distance = SeismicDistanceCalculator.calculate_haversine_distance(self.setup.site_lat[j], self.setup.site_lon[j], source_info['lat'], source_info['lon'])
#                 else:
#                     raise ValueError("Invalid distance type. Choose 'rupture', 'joyner_boore', 'horizontal', or 'haversine'.")
#
#                 # Append distances for plotting
#                 distance_data.append((self.setup.site_lon[j], self.setup.site_lat[j], distance))
#
#             # Prepare the data
#             lons, lats, distances = zip(*distance_data)
#
#             # Generate a grid to interpolate onto
#             grid_lons, grid_lats = np.meshgrid(np.linspace(min(lons), max(lons), 100),
#                                                np.linspace(min(lats), max(lats), 100))
#
#             # Interpolate the data onto the grid
#             grid_distances = griddata((lons, lats), distances, (grid_lons, grid_lats), method='cubic')
#
#             # Create the contour plot
#             plt.figure(figsize=(10, 6))
#             contour = plt.contourf(grid_lons, grid_lats, grid_distances, levels=100, cmap='viridis')
#             colorbar = plt.colorbar(contour, label='Distance (km)')
#
#             # Optional: plot seismic sources if needed
#             for source_id, source_info in self.setup.source_data.items():
#                 plt.plot(source_info['lon'], source_info['lat'], '*', color='yellow', markersize=15, label=f'{source_id}')
#
#             # Annotations and titles
#             plt.title(f"{distance_type.capitalize()} Distance Contour Map")
#             plt.xlabel('Longitude')
#             plt.ylabel('Latitude')
#             plt.legend()
#             plt.show()
#
#     def plot_im_contour_map(self, exceedance_probability, ground_motion_type):
#         """
#         Plot a contour map representing seismic hazard based on the exceedance probability.
#
#         Parameters:
#         -----------
#         exceedance_probability : float
#             The exceedance probability for which to generate the contour map.
#         ground_motion_type : str
#             The type of ground motion ('PGA' or 'PGV') to prepare data for.
#         """
#         # Prepare the data
#         data = self.prepare_im_data(exceedance_probability, ground_motion_type)
#         lons, lats, ims = zip(*data)
#
#         # Generate a grid to interpolate onto
#         grid_lons, grid_lats = np.meshgrid(np.linspace(min(lons), max(lons), 100),
#                                           np.linspace(min(lats), max(lats), 100))
#
#         # Interpolate the data onto the grid
#         grid_ims = griddata((lons, lats), ims, (grid_lons, grid_lats), method='cubic')
#
#         # Determine unit based on ground motion type
#         if ground_motion_type == 'PGA':
#             unit = 'g'
#         elif ground_motion_type == 'PGV':
#             unit = 'cm/s'
#         else:
#             unit = ''  # Default if neither PGA nor PGV
#
#         # Create the contour plot
#         plt.figure(figsize=(10, 6))
#         contour = plt.contourf(grid_lons, grid_lats, grid_ims, levels=100, cmap='viridis')
#         colorbar = plt.colorbar(contour, label=f'{ground_motion_type} [{unit}]')
#
#         # Optional: plot seismic sources if needed
#         for source_id, source_info in self.setup.source_data.items():
#             plt.plot(source_info['lon'], source_info['lat'], '*', color='yellow', markersize=15, label=f'{source_id}')
#
#         # Annotations and titles
#         plt.title(f"Seismic Hazard Map - {self.return_period} years, {exceedance_probability*100}% Exceedance Probability")
#         plt.xlabel('Longitude')
#         plt.ylabel('Latitude')
#         plt.legend()
#         plt.show()
#
#     def plot_tract_intensity_map(self, census_file_path, exceedance_probability, ground_motion_type):
#         """
#         Plot a map representing seismic hazard with intensity measure values averaged for each census tract.
#
#         Parameters:
#         -----------
#         exceedance_probability : float
#             The exceedance probability for which to generate the contour map.
#         ground_motion_type : str
#             The type of ground motion ('PGA' or 'PGV') used for data preparation.
#         census_file_path : str
#             The path to the shapefile for North America.
#         """
#         # Prepare the intensity measure data
#         data = self.prepare_im_data(exceedance_probability, ground_motion_type)
#         lons, lats, ims = zip(*data)
#
#         # Create a GeoDataFrame from the intensity measure data
#         points_df = pd.DataFrame({'intensity_measure': ims, 'geometry': [Point(lon, lat) for lon, lat in zip(lons, lats)]})
#         points_gdf = gpd.GeoDataFrame(points_df, geometry='geometry')
#
#         # Read the shapefile and ensure points_gdf has the same CRS as the shapefile data
#         tracts_gdf = gpd.read_file(census_file_path)
#         points_gdf.crs = tracts_gdf.crs
#
#         # Perform a spatial join between the points and the tracts
#         tracts_with_ims = gpd.sjoin(tracts_gdf, points_gdf, how="inner", predicate='contains')
#
#         # Compute the mean intensity measure for each tract
#         tract_intensity = tracts_with_ims.groupby('TRACTCE')['intensity_measure'].mean().reset_index()
#
#         # Merge this back with the original tracts data
#         tracts_gdf = tracts_gdf.merge(tract_intensity, on='TRACTCE', how='left')
#
#         # Determine unit based on ground motion type
#         if ground_motion_type == 'PGA':
#             unit = 'g'
#         elif ground_motion_type == 'PGV':
#             unit = 'cm/s'
#         else:
#             unit = ''  # Default if neither PGA nor PGV
#
#         # Plot the tracts colored by the mean intensity measure
#         fig, ax = plt.subplots(1, 1, figsize=(10, 6))
#         tracts_gdf.plot(column='intensity_measure', ax=ax, legend=True, cmap='viridis', legend_kwds={'label': f'{ground_motion_type} ({unit})'}, missing_kwds={
#             "color": "lightgrey",
#             "edgecolor": "red",
#             "hatch": "///",
#             "label": "Missing values"
#         })
#
#         # Plot the seismic sources as yellow stars
#         for source_info in self.setup.source_data.values():
#             ax.plot(source_info['lon'], source_info['lat'], '*', color='yellow', markersize=15, label='Seismic Sources')
#
#         # Annotations and titles
#         plt.title(f"Seismic Hazard Map - {self.return_period} years, {exceedance_probability*100}% Exceedance Probability")
#         plt.xlabel('Longitude')
#         plt.ylabel('Latitude')
#         plt.legend()
#         plt.show()
