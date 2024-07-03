# Copyright (c) 2019 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/


import concurrent.futures
from itertools import repeat

from pyincore import AnalysisUtil, GeoUtil
from pyincore import BaseAnalysis, HazardService, FragilityService
from pyincore.analyses.bridgedamage.bridgeutil import BridgeUtil
from pyincore.models.dfr3curve import DFR3Curve

from pyincore import PortComponentDamage  # TODO: Check import


class YardDamage(PortComponentDamage):
    """Computes bridge structural damage for earthquake, tsunami, tornado, and hurricane hazards.

    Args:
        incore_client (IncoreClient): Service authentication.

    """

    def __init__(self, incore_client):
        self.hazardsvc = HazardService(incore_client)
        self.fragilitysvc = FragilityService(incore_client)

        super(YardDamage, self).__init__(incore_client)

    # Keep get_spec in child classess

    def get_spec(self):
        """Get specifications of the bridge damage analysis.

        Returns:
            obj: A JSON object of specifications of the bridge damage analysis.

        """
        return {
            "name": "bridge-damage",
            "description": "bridge damage analysis",
            "input_parameters": [
                {
                    "id": "result_name",
                    "required": True,
                    "description": "result dataset name",
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
                    "type": ["earthquake", "tornado", "hurricane", "flood", "tsunami"],
                },
            ],
            "input_datasets": [
                {
                    "id": "cranes",
                    "required": True,
                    "description": "Bridge Inventory",
                    "type": ["ergo:cranes", "ergo:bridgesVer2", "ergo:bridgesVer3"],
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
                    "parent_type": "cranes",
                    "description": "CSV file of bridge structural damage",
                    "type": "ergo:bridgeDamageVer3",
                },
                {
                    "id": "metadata",
                    "parent_type": "cranes",
                    "description": "additional metadata in json file about applied hazard value and "
                    "fragility",
                    "type": "incore:bridgeDamageSupplement",
                },
            ],
        }
