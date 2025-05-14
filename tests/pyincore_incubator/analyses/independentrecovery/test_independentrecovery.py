from pyincore import IncoreClient, Dataset
from pyincore_incubator.analyses.independentrecovery import IndependentRecovery
import pyincore.globals as pyglobals


def run_with_base_class():
    # Connect to IN-CORE service
    client = IncoreClient(pyglobals.INCORE_API_DEV_URL)
    client.clear_cache()

    independent_recovery = IndependentRecovery(client)

    building_damage = Dataset.from_file(
        "data/Joplin_bldg_dmg_result.csv", "ergo:buildingDamageVer4"
    )
    household_inventory = Dataset.from_file(
        "data/hui_ver2-0-0_Joplin_MO_2010_rs1000.csv", "incore:housingUnitInventory"
    )
    household_allocation = Dataset.from_file(
        "data/Joplin_hua_result_1238.csv", "incore:housingUnitAllocation"
    )
    household_dislocation = Dataset.from_file(
        "data/pop-dislocation-results.csv", "incore:popDislocation"
    )
    residential_recovery = Dataset.from_file(
        "data/joplin_recovery_sample10_recovery.csv", "incore:buildingRecovery"
    )
    housing_recovery = Dataset.from_file(
        "data/housing_recovery_result.csv", "incore:housingRecoveryHistory"
    )

    independent_recovery.set_input_dataset("building_damage", building_damage)
    independent_recovery.set_input_dataset(
        "housing_unit_inventory", household_inventory
    )
    independent_recovery.set_input_dataset(
        "housing_unit_allocation", household_allocation
    )
    independent_recovery.set_input_dataset(
        "population_dislocation", household_dislocation
    )
    independent_recovery.set_input_dataset("residential_recovery", residential_recovery)
    independent_recovery.set_input_dataset(
        "household_housing_recovery", housing_recovery
    )

    # Specify the result name
    result_name = "independent_recovery"

    # Set analysis parameters
    independent_recovery.set_parameter("result_name", result_name)

    # Run the analysis (NOTE: with SettingWithCopyWarning)
    independent_recovery.run_analysis()


if __name__ == "__main__":
    run_with_base_class()
