# Copyright (c) 2024 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import pygmm

from pyincore import BaseAnalysis, Dataset
from pyincore_incubator.analyses.earthquakegenerator.seismic_distance_calculator import (
    SeismicDistanceCalculator,
)
from pyincore_incubator.analyses.earthquakegenerator.initial_setup import InitialSetup
from pyincore_incubator.analyses.earthquakegenerator.seismic_hazard_visualization import (
    SeismicHazardVisualization,
)
from pyincore_incubator.analyses.earthquakegenerator.magnitude_frequency_distribution import (
    MagnitudeFrequencyDistribution,
)


class EarthquakeGenerator(BaseAnalysis):
    """This Analysis computes a probabilistic earthquake hazard by generating earthquake scenarios.

    The output is a dataset in tif format

    Contributors
        | Science: Zeinab Farahmandfar
        | Implementation: Zeinab Farahmandfar, Christopher Navarro

    """

    def __init__(self, incore_client):
        super(EarthquakeGenerator, self).__init__(incore_client)

    def run(self):
        """Executes seismic hazard analysis."""

        inventory_dataset = self.get_input_dataset("inventory")
        inventory_df = inventory_dataset.get_dataframe_from_shapefile()

        # TODO these values could come from a dataset that has site information
        if "condition" not in inventory_df:
            if self.get_parameter("default_site_condition") is not None:
                inventory_df["condition"] = self.get_parameter("default_site_condition")
            else:
                inventory_df["condition"] = "soil"
        if "vs30" not in inventory_df:
            if self.get_parameter("default_site_vs30") is not None:
                inventory_df["vs30"] = self.get_parameter("default_site_vs30")
            else:
                inventory_df["vs30"] = 250
        if "depth" not in inventory_df:
            if self.get_parameter("default_site_depth") is not None:
                inventory_df["depth"] = self.get_parameter("default_site_depth")
            else:
                inventory_df["depth"] = 0

        source_dataset = self.get_input_dataset("earthquake_data")
        eq_json = source_dataset.get_json_reader()
        setup = InitialSetup(inventory_df, eq_json)

        (
            intensity_measures_dict,
            intensity_measures_df,
        ) = self.generate_earthquake_objects(setup)

        return_period = self.get_parameter("return_period")
        num_scenarios = self.get_parameter("num_scenarios")
        ground_motion_type = self.get_parameter("ground_motion_type")

        seismic_hazard_visualization = SeismicHazardVisualization(
            setup, intensity_measures_dict, num_scenarios, return_period=return_period
        )

        exceedance_probability = self.get_parameter("exceedance_probability")
        grid_resolution = 0.01
        if self.get_parameter("grid_resolution") is not None:
            grid_resolution = self.get_parameter("grid_resolution")

        result_name = self.get_parameter("result_name")
        output_file_name = f"{result_name}_Continuous_{ground_motion_type}.tif"
        seismic_hazard_visualization.generate_continuous_earthquake_geotiff(
            exceedance_probability=exceedance_probability,
            ground_motion_type=ground_motion_type,
            grid_resolution=grid_resolution,
            geotiff_file_name=output_file_name,
        )

        self.set_output_dataset(
            "result",
            Dataset.from_file(
                output_file_name,
                data_type=self.output_datasets["result"]["spec"]["type"],
            ),
        )

        # TODO demand units should be set by the ground motion type
        metadata = {}
        metadata["hazardType"] = "probabilistic"
        metadata["demandType"] = ground_motion_type
        if ground_motion_type == "PGA" or ground_motion_type == "SA":
            metadata["demandUnits"] = "g"
        elif ground_motion_type == "PGV":
            metadata["demandUnits"] = "cm/s"

        if self.get_parameter("period") is not None:
            metadata["period"] = self.get_parameter("period")

        metadata["exceedance_probability"] = exceedance_probability

        # Create the result dataset
        self.set_result_json_data(
            "metadata",
            metadata,
            name=self.get_parameter("result_name") + "_additional_info",
        )

        return True

    def generate_scenarios(self, setup, num_scenarios):
        """
        Generates earthquake scenarios with associated magnitudes and sources.

        Parameters:
        -----------
        num_scenarios : int
            Number of earthquake scenarios to generate.

        Returns:
        --------
        tuple: A tuple containing arrays of magnitudes and corresponding source names.
        """
        # Generate magnitudes for each source within their specified range.
        source_magnitudes = {
            source_name: np.linspace(
                setup.source_m_min[source_name],
                setup.source_m_max[source_name],
                num_scenarios,
            )
            for source_name in setup.source_data
        }

        # Flatten the source magnitudes into a single array
        all_magnitudes = np.concatenate(
            [source_magnitudes[source_name] for source_name in setup.source_data]
        )

        magnitude_frequency_distribution_class = self.get_parameter(
            "magnitude_frequency_distribution"
        )

        magnitude_frequency_distribution = getattr(
            MagnitudeFrequencyDistribution, magnitude_frequency_distribution_class
        )

        # Calculate the probability of each magnitude for each source.
        probabilities = {
            source_name: magnitude_frequency_distribution(
                source_magnitudes[source_name],
                setup.source_m_min[source_name],
                setup.source_m_max[source_name],
            )[0]
            for source_name in setup.source_data
        }

        # Sum the probabilities across all sources, weighted by the annual rate of exceedance.
        prob_of_magnitudes = np.sum(
            [
                setup.source_nu[source_name] * probabilities[source_name]
                for source_name in setup.source_data
            ],
            axis=0,
        )

        # Normalize the probabilities to sum to 1.
        prob_of_magnitudes /= np.sum(prob_of_magnitudes)

        # Expand probabilities to match the size of the concatenated magnitudes array
        expanded_prob_of_magnitudes = np.tile(
            prob_of_magnitudes, len(setup.source_data)
        )

        # Normalize the expanded probabilities to sum to 1.
        expanded_prob_of_magnitudes /= np.sum(expanded_prob_of_magnitudes)

        # Select magnitudes based on the calculated probabilities.
        magnitudes = np.random.choice(
            all_magnitudes, size=num_scenarios, p=expanded_prob_of_magnitudes
        )

        # Calculate the probability of each source given the selected magnitudes.
        prob_of_magnitudes_given_source = {
            source_name: magnitude_frequency_distribution(
                magnitudes,
                setup.source_m_min[source_name],
                setup.source_m_max[source_name],
            )[0]
            for source_name in setup.source_data
        }

        # Normalize the probabilities of sources given the magnitude.
        prob_of_sources_given_magnitude = np.transpose(
            [
                setup.source_nu[source_name]
                * prob_of_magnitudes_given_source[source_name]
                / sum(
                    setup.source_nu[source_name]
                    * prob_of_magnitudes_given_source[source_name]
                    for source_name in setup.source_data
                )
                for source_name in setup.source_data
            ]
        )

        # Select sources for each scenario based on the calculated probabilities.
        sources = [
            list(setup.source_data.keys())[
                np.random.choice(range(setup.num_sources), p=prob)
            ]
            for prob in prob_of_sources_given_magnitude
        ]

        return magnitudes, sources

    def calculate_ground_motion_parameters(
        self, setup, ground_motion_type, magnitudes, sources
    ):
        """
        Calculates the median ground motion parameters and the associated standard deviations.

        Returns:
        --------
        tuple: A tuple containing dictionaries for median ground motion, intra-event std, and inter-event std.
        """

        def _get_gm_indices(model_class, ground_motion_type):
            if ground_motion_type == "PGA":
                return model_class.INDEX_PGA
            elif ground_motion_type == "PGV":
                return model_class.INDEX_PGV
            elif ground_motion_type == "SA":
                return model_class.INDICES_PSA
            else:
                raise ValueError(
                    "Invalid ground motion type. Choose 'PGA', 'PGV', or 'SA'."
                )

        median_ground_motion, intra_event_std, inter_event_std = {}, {}, {}

        ground_motion_equation_model = self.get_parameter(
            "ground_motion_equation_model"
        )

        for i, magnitude in enumerate(magnitudes):
            source_info = setup.source_data[sources[i]]
            source_loc = (source_info["lon"], source_info["lat"], source_info["depth"])
            # source_loc = (source_info["lat"], source_info["lon"], source_info["depth"])

            for j, site_id in enumerate(setup.site_id):
                site_loc = (
                    setup.site_lon[j],
                    setup.site_lat[j],
                    setup.site_depth[j],
                )

                rupture_distance = SeismicDistanceCalculator.calculate_rupture_distance(
                    site_loc, source_loc, source_info["strike"], source_info["dip"]
                )
                joyner_boore_distance = (
                    SeismicDistanceCalculator.calculate_joyner_boore_distance(
                        site_loc, source_loc
                    )
                )
                horizontal_distance = (
                    SeismicDistanceCalculator.calculate_horizontal_distance(
                        site_loc, source_loc, source_info["strike"]
                    )
                )

                # print("distances")
                # print(rupture_distance)
                # print(joyner_boore_distance)
                # print(horizontal_distance)

                model_class = getattr(pygmm, ground_motion_equation_model)
                gm_index = _get_gm_indices(model_class, ground_motion_type)

                scenario = pygmm.Scenario(
                    mag=magnitude,
                    dist_rup=rupture_distance,
                    dist_jb=joyner_boore_distance,
                    dist_x=horizontal_distance,
                    site_cond=setup.site_condition[j],
                    v_s30=setup.site_vs30[j],
                    dip=source_info["dip"],
                    mechanism=source_info["mechanism"],
                    event_type=source_info["event_type"],
                )
                model = model_class(scenario)

                resp_ref = np.exp(model._calc_ln_resp(model.V_REF, np.nan))
                ln_std, tau, phi = model._calc_ln_std(resp_ref)

                median_ground_motion[site_id, i] = np.exp(
                    model._calc_ln_resp(scenario.v_s30, resp_ref)[gm_index]
                )
                inter_event_std[site_id, i] = tau[gm_index]
                intra_event_std[site_id, i] = phi[gm_index]

        return median_ground_motion, intra_event_std, inter_event_std

    def generate_norm_inter_event_residuals(
        self, setup, num_scenarios, inter_event_std
    ):
        """
        Generates a set of normalized inter-event residuals for each scenario.

        Parameters:
        -----------
        inter_event_std : dict
            A dictionary with (site_id, scenario_index) as keys and standard deviation of inter-event residuals as values.

        Returns:
        --------
        dict: A dictionary with (site_id, scenario_index) as keys and normalized inter-event residuals as values.
        """

        norm_inter_event_residuals = {}
        # Generate normalized inter-event residuals for each scenario.
        for i in range(num_scenarios):
            Eta = np.random.normal(0, 1)
            for j, site_id in enumerate(setup.site_id):
                # Base standard deviation for normalization
                base_inter_event_std = inter_event_std[setup.site_id[0], i]
                norm_inter_event_residuals[site_id, i] = (
                    base_inter_event_std / inter_event_std[site_id, i]
                ) * Eta

        return norm_inter_event_residuals

    def generate_norm_intra_event_residuals(
        self, setup, num_scenarios, ground_motion_type, sources, period=None
    ):
        """
        Generates a set of intra-event residuals using a spatial correlation model.

        Returns:
        --------
        dict: A dictionary with (site_id, scenario_index) as keys and intra-event residuals as values.
        """

        def determine_range_parameter_by_ground_motion_type(
            vs30, ground_motion_type, period=None
        ):
            """
            Determine the range parameter (b) for the correlation model given Vs30 values,
            the ground motion type (PGA, PGV, or SA), and period if applicable.

            Parameters:
            -----------
            vs30 : (pd.Series)
                A Pandas series containing Vs30 values at different locations.
            ground_motion_type : str
                Type of ground motion to analyze ('PGA', 'PGV', or 'SA').
            period : float, optional
                The period of interest in seconds for spectral acceleration (SA).

            Returns:
            --------
            float: The range parameter b for the correlation model.
            """
            if isinstance(vs30, (list, np.ndarray, pd.Series)) and len(vs30) > 1:
                std_vs30 = np.std(vs30)
            else:
                std_vs30 = 0

            clustering_threshold = 50

            if ground_motion_type == "PGA":
                period = 0
                case = 1 if std_vs30 > clustering_threshold else 2

                if case == 1:
                    b = 8.5 + 17.2 * period
                else:
                    b = 40.7 - 15.0 * period

            elif ground_motion_type == "PGV":
                period = 1
                b = 25.7

            elif ground_motion_type == "SA":
                if period is None:
                    raise ValueError(
                        "Period must be provided for Spectral Acceleration (SA)."
                    )

                case = 1 if std_vs30 > clustering_threshold else 2

                if period < 1:
                    if case == 1:
                        b = 8.5 + 17.2 * period
                    else:
                        b = 40.7 - 15.0 * period
                else:
                    b = 22.0 + 3.7 * period

            else:
                raise ValueError(
                    "Invalid ground motion type. Choose 'PGA', 'PGV', or 'SA'."
                )
            return b

        # Range parameter b based on ground motion type.
        b = determine_range_parameter_by_ground_motion_type(
            setup.site_vs30, ground_motion_type, period
        )

        # Coordinate matrix for sites.
        coords = np.column_stack((setup.site_lat, setup.site_lon))

        # Calculate the pairwise Haversine distance matrix for the current scenario's source location.
        distances = pdist(
            coords,
            lambda u, v: SeismicDistanceCalculator.calculate_haversine_distance(
                u[0], u[1], v[0], v[1]
            ),
        )

        # Convert distance matrix to a square-form and calculate the correlation matrix.
        corr_matrix = np.exp(-3 * squareform(distances) / b)

        # Set the diagonal to 1 to ensure positive definiteness.
        np.fill_diagonal(corr_matrix, 1)

        norm_intra_event_residuals = {}
        # Generate residuals for each scenario.
        for i in range(num_scenarios):
            # Generate the intra-event residuals for this scenario using the correlation matrix.
            epsilon = np.random.multivariate_normal(
                np.zeros(setup.num_sites), corr_matrix
            )

            # Assign residuals to each site for the current scenario.
            for j, site_id in enumerate(setup.site_id):
                norm_intra_event_residuals[(site_id, i)] = epsilon[j]

        return norm_intra_event_residuals

    def generate_earthquake_objects(self, setup):
        """
        Generates a dictionary of earthquake objects with calculated intensity measures.

        Returns:
        --------
        dict: A dictionary with tuples of (site_id, scenario_index) as keys and intensity measure values as values.
        """
        num_scenarios = self.get_parameter("num_scenarios")
        ground_motion_type = self.get_parameter("ground_motion_type")
        magnitudes, sources = self.generate_scenarios(setup, num_scenarios)
        period = self.get_parameter("period")

        intensity_measure_dict = {}
        intensity_measure_data = []
        combined_data = {}

        (
            median_ground_motion,
            intra_event_std,
            inter_event_std,
        ) = self.calculate_ground_motion_parameters(
            setup, ground_motion_type, magnitudes, sources
        )
        norm_inter_event_residual = self.generate_norm_inter_event_residuals(
            setup, num_scenarios, inter_event_std
        )
        norm_intra_event_residual = self.generate_norm_intra_event_residuals(
            setup, num_scenarios, ground_motion_type, sources, period
        )

        for i, magnitude in enumerate(magnitudes):
            source_info = setup.source_data[sources[i]]
            # source_loc = (
            #     source_info["lon"],
            #     source_info["lat"],
            #     source_info["depth"],
            # )
            source_id = sources[i]
            for j, site_id in enumerate(setup.site_id):
                # site_loc = (
                #     setup.site_lon[j],
                #     setup.site_lat[j],
                #     setup.site_depth[j],
                # )
                ln_im_value = (
                    np.log(median_ground_motion[site_id, i])
                    + inter_event_std[site_id, i]
                    * norm_inter_event_residual[site_id, i]
                    + intra_event_std[site_id, i]
                    * norm_intra_event_residual[site_id, i]
                )

                distance = SeismicDistanceCalculator.calculate_haversine_distance(
                    setup.site_lat[j],
                    setup.site_lon[j],
                    source_info["lat"],
                    source_info["lon"],
                )

                key = (i, site_id, source_id, magnitude, distance)
                if key not in combined_data:
                    combined_data[key] = {}
                combined_data[key][ground_motion_type] = np.exp(ln_im_value)

        # Transform combined data into the final dictionary and DataFrame format
        for key, values in combined_data.items():
            scenario_index, site_id, source_id, magnitude, distance = key
            intensity_measure_dict[key] = values
            intensity_measure_data.append(
                [
                    scenario_index,
                    site_id,
                    source_id,
                    magnitude,
                    distance,
                    values.get(ground_motion_type, None),
                ]
            )

        # Create a DataFrame from the intensity measures
        intensity_measure_df = pd.DataFrame(
            intensity_measure_data,
            columns=[
                "scenario_index",
                "site_id",
                "source_id",
                "magnitude",
                "distance",
                ground_motion_type,
            ],
        )

        return intensity_measure_dict, intensity_measure_df

    def get_spec(self):
        """Get specifications of the probabilistic earthquake generator analysis.

        Returns:
            obj: A JSON object of specifications of the probabilistic earthquake generator analysis.

        """
        return {
            "name": "earthquake-generator",
            "description": "example analysis",
            "input_parameters": [
                {
                    "id": "result_name",
                    "required": True,
                    "description": "result dataset name",
                    "type": str,
                },
                {
                    "id": "ground_motion_equation_model",
                    "required": True,
                    "description": "Ground motion model to use",
                    "type": str,
                },
                {
                    "id": "ground_motion_type",
                    "required": True,
                    "description": "Ground motion type (e.g. PGA, SA, etc)",
                    "type": str,
                },
                {
                    "id": "period",
                    "required": False,
                    "description": "The period of interest in seconds for spectral acceleration (SA)",
                    "type": float,
                },
                {
                    "id": "return_period",
                    "required": True,
                    "description": "The return period for seismic hazard analysis in years.",
                    "type": int,
                },
                {
                    "id": "exceedance_probability",
                    "required": True,
                    "description": "The exceedance probability for which to prepare the data (e.g. 0.05 for 5%)",
                    "type": float,
                },
                {
                    "id": "grid_resolution",
                    "required": False,
                    "description": "The resolution of the grid in degrees. Default is 0.01 degrees.",
                    "type": float,
                },
                {
                    "id": "num_scenarios",
                    "required": True,
                    "description": "Number of earthquake scenarios to generate.",
                    "type": int,
                },
                {
                    "id": "magnitude_frequency_distribution",
                    "required": True,
                    "description": "Name of the magnitude frequency distribution function to use.",
                    "type": str,
                },
                {
                    "id": "default_site_vs30",
                    "required": False,
                    "description": "",
                    "type": float,
                },
                {
                    "id": "default_site_depth",
                    "required": False,
                    "description": "",
                    "type": float,
                },
                {
                    "id": "default_site_condition",
                    "required": False,
                    "description": "",
                    "type": str,
                },
            ],
            "input_datasets": [
                {
                    "id": "inventory",
                    "required": True,
                    "description": "Inventory dataset",
                    "type": [
                        "ergo:buildingInventoryVer4",
                        "ergo:buildingInventoryVer5",
                        "ergo:buildingInventoryVer6",
                        "ergo:buildingInventoryVer7",
                        "ergo:bridges",
                        "ergo:epf",
                        "incore:epf",
                        "incore:epfVer2",
                        "ergo:waterFacilityTopo",
                    ],
                },
                {
                    "id": "earthquake_data",
                    "required": True,
                    "description": "Earthquake information",
                    "type": ["incore:eqData"],
                },
            ],
            "output_datasets": [
                {
                    "id": "result",
                    "parent_type": "",
                    "description": "probabilistic earthquake raster",
                    "type": "ergo:probabilisticEarthquakeRaster",
                },
                {
                    "id": "metadata",
                    "parent_type": "",
                    "description": "Json file with information about the generated hazard",
                    "type": "incore:hazardSupplement",
                },
            ],
        }
