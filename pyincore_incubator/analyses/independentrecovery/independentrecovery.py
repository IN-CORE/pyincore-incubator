# Copyright (c) 2024 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import numpy as np
import pandas as pd
import time

from pyincore import BaseAnalysis


class IndependentRecovery(BaseAnalysis):
    """
    This analysis computes the household housing recovery time considering the availability of their residential
     buildings and housing vacancies in the community. Currently, supported hazards are tornadoes.

    The methodology examines the interdependent community recovery process across physical infrastructure and social
     systems

    The outputs of this analysis is a CSV file with the modified history of housing recovery changes and the
     calculated housing housing recovery time.

    Contributors
        | Science: Wanting Lisa Wang, John W. van de Lindt, Elaina J. Sutley, and Sara Hamideh
        | Implementation: Wanting Lisa Wang, and NCSA IN-CORE Dev Team

    Related publications
        Wang, W., van de Lindt, J.W., Sutley, E. and Hamideh, S. "Interdependent Recovery Methodology for
         Residential Buildings and Household Housing in Community Resilience Modeling."
         ASCE OPEN: Multidisciplinary Journal of Civil Engineering (accepted).

    Args:
        incore_client (IncoreClient): Service authentication.

    """

    def __init__(self, incore_client):
        super(IndependentRecovery, self).__init__(incore_client)

    def run(self):
        """Executes the independent recovery analysis.

        Returns:
            bool: True if successful, False otherwise.

        """
        # TODO: Start using seed
        result_name = self.get_parameter("result_name")

        building_damage = self.get_input_dataset(
            "building damage"
        ).get_dataframe_from_csv(low_memory=False)
        household_inventory = self.get_input_dataset(
            "household inventory"
        ).get_dataframe_from_csv(low_memory=False)
        household_allocation = self.get_input_dataset(
            "household allocation"
        ).get_dataframe_from_csv(low_memory=False)
        household_dislocation = self.get_input_dataset(
            "household dislocation"
        ).get_dataframe_from_csv(low_memory=False)
        residential_recovery = self.get_input_dataset(
            "residential recovery"
        ).get_dataframe_from_csv(low_memory=False)
        housing_recovery = self.get_input_dataset(
            "household housing recovery"
        ).get_dataframe_from_csv(low_memory=False)

        # Returns dataframe
        recovery_results = self.independent_recovery(
            building_damage,
            household_inventory,
            household_allocation,
            household_dislocation,
            residential_recovery,
            housing_recovery,
        )
        self.set_result_csv_data(
            "housing recovery", recovery_results, result_name, "dataframe"
        )

        return True

    def independent_recovery(
        self,
        building_damage,
        household_inventory,
        household_allocation,
        household_dislocation,
        residential_recovery,
        housing_recovery,
    ):
        """
        Calculates independent recovery of residential buildings and household housing

        Args:
            building_damage (pd.DataFrame): Building damage results
            household_inventory (pd.DataFrame): Household inventory
            household_allocation (pd.DataFrame): Household allocation results
            household_dislocation (pd.DataFrame): Household dislocation results
            residential_recovery (pd.DataFrame): Residential building recovery results
            housing_recovery (pd.DataFrame): Household housing recovery results

        Returns:
            pd.DataFrame: household housing recovery sequences and time

        """

        start_independent_recovery = time.process_time()

        merged_super = self.merge_damage_recovery_housing_building(
            building_damage,
            household_inventory,
            household_dislocation,
            residential_recovery,
            housing_recovery,
        )

        modified_housing_recovery = self.modified_housing_recovery(
            merged_super, household_allocation
        )

        housing_recovery_time = self.housing_recovery_time(modified_housing_recovery)

        end_start_independent_recovery = time.process_time()
        print(
            "Finished independent_recovery() in "
            + str(end_start_independent_recovery - start_independent_recovery)
            + " secs"
        )

        result = housing_recovery_time

        return result

    @staticmethod
    def merge_damage_recovery_housing_building(
        building_damage,
        household_inventory,
        household_dislocation,
        residential_recovery,
        housing_recovery,
    ):
        """
        Load CSV files to pandas Dataframes, merge them and drop unused columns.

        Args:
            building_damage (pd.DataFrame): Building damage results
            household_inventory (pd.DataFrame): Household inventory
            household_dislocation (pd.DataFrame): Household dislocation results
            residential_recovery (pd.DataFrame): Residential building recovery results
            housing_recovery (pd.DataFrame): Household housing recovery results

        Returns:
            pd.DataFrame: A merged table of all the inputs.

        """

        building_damage = building_damage.loc[(building_damage["haz_expose"] == "yes")]
        household_dislocation = household_dislocation.loc[
            (household_dislocation["plcname10"] == "Joplin")
            & (household_dislocation["guid"].notnull())
            & (household_dislocation["huid"].notna())
        ]

        residential_recovery["bldgmean"] = round(
            residential_recovery.drop(columns=["guid"]).mean(axis=1), 2
        )

        housing_unit_inv_df = pd.merge(
            left=household_inventory,
            right=household_dislocation[
                ["huid", "guid", "DS_0", "DS_1", "DS_2", "DS_3", "prdis", "dislocated"]
            ],
            left_on="huid",
            right_on="huid",
            how="right",
        )
        housing_unit_inv_df = pd.merge(
            left=housing_unit_inv_df,
            right=building_damage["guid"],
            left_on="guid",
            right_on="guid",
            how="inner",
        )
        housing_unit_inv_df = pd.merge(
            left=housing_unit_inv_df,
            right=housing_recovery.drop("guid", axis=1),
            left_on="huid",
            right_on="huid",
            how="left",
        )
        household = pd.merge(
            left=housing_unit_inv_df,
            right=residential_recovery[["guid", "bldgmean"]],
            left_on="guid",
            right_on="guid",
            how="inner",
        )

        return household

    @staticmethod
    def modified_housing_recovery(merged_super, household_allocation):
        """Gets the modified housing unit sequences.

        Args:
            merged_super (pd.DataFrame): A merged table of all the inputs described above
            household_allocation (pd.DataFrame): Household allocation results

        Returns:
            pd.DataFrame: Results of the modified housing unit sequences considering the availability of their
            residential buildings and housing vacancies.
        """

        household_allocation = household_allocation.loc[
            (household_allocation["plcname10"] == "Joplin")
            & (household_allocation["guid"].notnull())
            & (household_allocation["huid"].notna())
        ]

        for j in range(0, len(merged_super)):
            rand = np.random.random()
            for i in range(0, 90):
                if (
                    (merged_super.iloc[j, -91 + i] == 4)
                    and (4 * (i + 1) < merged_super.iloc[j, -91 + 90])
                    and rand
                    > household_allocation["vacancy"].astype(bool).sum(axis=0)
                    / merged_super["ownershp"].count()
                ):
                    merged_super.iloc[j, i + 4] = 3

        # Construct a new DataFrame
        modified_housing_recovery_results = merged_super

        return modified_housing_recovery_results

    @staticmethod
    def housing_recovery_time(modified_housing_recovery):
        """Gets household housing recovery time.

        Args:
            modified_housing_recovery (pd.DataFrame): The modified housing unit sequences considering the availability
             of their residential buildings and housing vacancies.

        Returns:
            pd.DataFrame: Results of household housing recovery time
        """
        modified_housing_recovery["hur"] = np.nan
        modified_housing_recovery.insert(0, "hur", modified_housing_recovery.pop("hur"))
        for j in range(0, len(modified_housing_recovery)):
            for i in range(1, 90):
                if modified_housing_recovery.iloc[j, -90 + i] == 4:
                    modified_housing_recovery["hur"][j] = i * 4
                    break
                if modified_housing_recovery.iloc[j, -90 + i] == 5:
                    modified_housing_recovery["hur"][j] = 360
                    break

        housing_recovery_time_results = modified_housing_recovery

        return housing_recovery_time_results

    def get_spec(self):
        """Get specifications of the residential building recovery analysis.

        Returns:
            obj: A JSON object of specifications of the independent recovery analysis.

        """
        return {
            "name": "independent-recovery",
            "description": "calculate household housing recovery considering the availability of their residential "
            "buildings and housing vacancies",
            "input_parameters": [
                {
                    "id": "result_name",
                    "required": True,
                    "description": "name of the result",
                    "type": str,
                }
            ],
            "input_datasets": [
                {
                    "id": "building damage",
                    "required": True,
                    "description": "Building Damage Probability",
                    "type": ["ncsa:buildingDamageVer4"],
                },
                {
                    "id": "household inventory",
                    "required": True,
                    "description": "Household Inventory",
                    "type": ["incore:housingUnitInventory"],
                },
                {
                    "id": "household allocation",
                    "required": True,
                    "description": "Household Allocation",
                    "type": ["incore:housingUnitAllocation"],
                },
                {
                    "id": "household dislocation",
                    "required": True,
                    "description": "Household Dislocation",
                    "type": ["incore:householdDislocation"],
                },
                {
                    "id": "residential recovery",
                    "required": True,
                    "description": "Residential Building Recovery",
                    "type": ["incore:buildingRecovery"],
                },
                {
                    "id": "household housing recovery",
                    "required": True,
                    "description": "Household Housing Recovery",
                    "type": ["incore:housingRecovery"],
                },
            ],
            "output_datasets": [
                {
                    "id": "housing recovery",
                    "description": "CSV file of household housing recovery considering the availability of their"
                    " residential buildings and housing vacancies",
                    "type": "incore:housingRecovery",
                }
            ],
        }
