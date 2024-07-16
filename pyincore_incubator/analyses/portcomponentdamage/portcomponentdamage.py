# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/


import concurrent.futures
from itertools import repeat

from pyincore import AnalysisUtil, GeoUtil
from pyincore import BaseAnalysis, HazardService, FragilityService

# from pyincore.analyses.bridgedamage.bridgeutil import BridgeUtil
from pyincore.models.dfr3curve import DFR3Curve


class PortComponentDamage(BaseAnalysis):
    """Computes port component structural damage for hurricane hazards.

    Args:
        incore_client (IncoreClient): Service authentication.

    """

    def __init__(self, incore_client):
        self.hazardsvc = HazardService(incore_client)
        self.fragilitysvc = FragilityService(incore_client)

        super(PortComponentDamage, self).__init__(incore_client)

    def run(self):
        """Executes port component damage analysis."""
        # component dataset
        component_set = self.get_input_dataset("components").get_inventory_reader()

        # get input hazard
        hazard, hazard_type, hazard_dataset_id = (
            self.create_hazard_object_from_input_params()
        )

        user_defined_cpu = 1

        if (
            not self.get_parameter("num_cpu") is None
            and self.get_parameter("num_cpu") > 0
        ):
            user_defined_cpu = self.get_parameter("num_cpu")

        num_workers = AnalysisUtil.determine_parallelism_locally(
            self, len(component_set), user_defined_cpu
        )

        avg_bulk_input_size = int(len(component_set) / num_workers)
        inventory_args = []
        count = 0
        inventory_list = list(component_set)
        while count < len(inventory_list):
            inventory_args.append(inventory_list[count : count + avg_bulk_input_size])
            count += avg_bulk_input_size

        (ds_results, damage_results) = self.port_component_damage_concurrent_future(
            self.port_component_damage_analysis_bulk_input,
            num_workers,
            inventory_args,
            repeat(hazard),
            repeat(hazard_type),
            repeat(hazard_dataset_id),
        )

        self.set_result_csv_data(
            "result", ds_results, name=self.get_parameter("result_name")
        )
        self.set_result_json_data(
            "metadata",
            damage_results,
            name=self.get_parameter("result_name") + "_additional_info",
        )

        return True

    def port_component_damage_concurrent_future(
        self, function_name, num_workers, *args
    ):
        """Utilizes concurrent.future module.

        Args:
            function_name (function): The function to be parallelized.
            num_workers (int): Maximum number workers in parallelization.
            *args: All the arguments in order to pass into parameter function_name.

        Returns:
            list: A list of ordered dictionaries with component damage values and other data/metadata.

        """
        output_ds = []
        output_dmg = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            for ret1, ret2 in executor.map(function_name, *args):
                output_ds.extend(ret1)
                output_dmg.extend(ret2)

        return output_ds, output_dmg

    def port_component_damage_analysis_bulk_input(
        self, components, hazard, hazard_type, hazard_dataset_id
    ):
        """Run analysis for multiple components.

        Args:
            components (list): Multiple components from input inventory set.
            hazard (obj): Hazard object.
            hazard_type (str): Type of hazard.
            hazard_dataset_id (str): ID of hazard.

        Returns:
            list: A list of ordered dictionaries with component damage values and other data/metadata.

        """

        # Get Fragility key
        fragility_key = self.get_parameter("fragility_key")

        fragility_set = self.fragilitysvc.match_inventory(
            self.get_input_dataset("dfr3_mapping_set"), components, fragility_key
        )

        values_payload = []
        unmapped_components = []
        mapped_components = []
        for c in components:
            component_id = c["id"]
            if component_id in fragility_set:
                location = GeoUtil.get_location(c)
                loc = str(location.y) + "," + str(location.x)

                demands = fragility_set[component_id].demand_types
                units = fragility_set[component_id].demand_units
                value = {"demands": demands, "units": units, "loc": loc}
                values_payload.append(value)
                mapped_components.append(c)

            else:
                unmapped_components.append(c)

        # not needed anymore as they are already split into mapped and unmapped
        del components

        hazard_vals = hazard.read_hazard_values(values_payload, self.hazardsvc)

        ds_results = []
        damage_results = []

        i = 0
        for component in mapped_components:
            ds_result = dict()
            damage_result = dict()
            dmg_probability = dict()
            dmg_intervals = dict()
            selected_fragility_set = fragility_set[component["id"]]

            if isinstance(selected_fragility_set.fragility_curves[0], DFR3Curve):
                # Supports multiple demand types in same fragility
                hazard_val = AnalysisUtil.update_precision_of_lists(
                    hazard_vals[i]["hazardValues"]
                )
                input_demand_types = hazard_vals[i]["demands"]
                input_demand_units = hazard_vals[i]["units"]

                hval_dict = dict()
                j = 0
                for d in selected_fragility_set.demand_types:
                    hval_dict[d] = hazard_val[j]
                    j += 1

                if not AnalysisUtil.do_hazard_values_have_errors(
                    hazard_vals[i]["hazardValues"]
                ):
                    component_args = (
                        selected_fragility_set.construct_expression_args_from_inventory(
                            component
                        )
                    )
                    dmg_probability = selected_fragility_set.calculate_limit_state(
                        hval_dict, inventory_type="portComponent", **component_args
                    )  # TODO: check inventory type

                    dmg_intervals = selected_fragility_set.calculate_damage_interval(
                        dmg_probability,
                        hazard_type=hazard_type,
                        inventory_type="portComponent",
                    )
            else:
                raise ValueError(
                    "One of the fragilities is in deprecated format. This should not happen. If you are "
                    "seeing this please report the issue."
                )

            ds_result["guid"] = component["properties"]["guid"]
            ds_result.update(dmg_probability)
            ds_result.update(dmg_intervals)
            ds_result["haz_expose"] = AnalysisUtil.get_exposure_from_hazard_values(
                hazard_val, hazard_type
            )

            damage_result["guid"] = component["properties"]["guid"]
            damage_result["fragility_id"] = selected_fragility_set.id
            # damage_result["retrofit"] = retrofit_type
            # damage_result["retrocost"] = retrofit_cost
            damage_result["demandtypes"] = input_demand_types
            damage_result["demandunits"] = input_demand_units
            damage_result["hazardtype"] = hazard_type
            damage_result["hazardval"] = hazard_val

            ds_results.append(ds_result)
            damage_results.append(damage_result)
            i += 1

        for component in unmapped_components:
            ds_result = dict()
            damage_result = dict()

            ds_result["guid"] = component["properties"]["guid"]

            damage_result["guid"] = component["properties"]["guid"]
            damage_result["retrofit"] = None
            damage_result["retrocost"] = None
            damage_result["demandtypes"] = None
            damage_result["demandunits"] = None
            damage_result["hazardtype"] = None
            damage_result["hazardval"] = None
            damage_result["spans"] = None  # TODO: check what's needed

            ds_results.append(ds_result)
            damage_results.append(damage_result)

        return ds_results, damage_results

    def get_spec(self):
        """Get specifications of the port component damage analysis.

        Returns:
            obj: A JSON object of specifications of the component damage analysis.

        """
        return {
            "name": "port-component-damage",
            "description": "Port component damage analysis",
            "input_parameters": [
                {
                    "id": "result_name",
                    "required": True,
                    "description": "Result dataset name",
                    "type": str,
                },
                {
                    "id": "fragility_key",
                    "required": False,
                    "description": "Fragility key to use in mapping dataset",
                    "type": str,
                },
                {
                    "id": "use_liquefaction",
                    "required": False,
                    "description": "Use liquefaction",
                    "type": bool,
                },
                {
                    "id": "liquefaction_geology_dataset_id",
                    "required": False,
                    "description": "Geology dataset id",
                    "type": str,
                },
                {
                    "id": "use_hazard_uncertainty",
                    "required": False,
                    "description": "Use hazard uncertainty",
                    "type": bool,
                },
                {
                    "id": "num_cpu",
                    "required": False,
                    "description": "If using parallel execution, the number of cpus to request",
                    "type": int,
                },
                {
                    "id": "hazard_id",
                    "required": False,
                    "description": "Hazard object id",
                    "type": str,
                },
                {
                    "id": "hazard_type",
                    "required": False,
                    "description": "Hazards type",
                    "type": str,
                },
            ],
            "input_hazards": [
                {
                    "id": "hazard",
                    "required": False,
                    "description": "Hazard object",
                    "type": ["hurricane"],
                },  # "earthquake", "tornado", "hurricane", "flood", "tsunami"
            ],
            "input_datasets": [
                {
                    "id": "components",
                    "required": True,
                    "description": "Port component inventory",
                    "type": ["portComponents"],  # TODO: Check new type
                },
                {
                    "id": "dfr3_mapping_set",
                    "required": True,
                    "description": "DFR3 Mapping Set Object",
                    "type": ["incore:dfr3MappingSet"],
                },
            ],
            "output_datasets": [
                {
                    "id": "result",
                    "parent_type": "components",
                    "description": "CSV file of component structural damage",
                    "type": "ergo:componentDamageVer3",  # TODO: Check new type
                },
                {
                    "id": "metadata",
                    "parent_type": "components",
                    "description": "additional metadata in json file about applied hazard value and "
                    "fragility",
                    "type": "incore:componentDamageSupplement",
                },
            ],
        }
