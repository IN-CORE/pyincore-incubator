from pyincore import IncoreClient, RepairService, MappingSet, Dataset, DataService
from pyincore_incubator.analyses.independentrecovery import IndependentRecovery


def run_with_base_class():
    # Connect to IN-CORE service
    client = IncoreClient()
    client.clear_cache()

    independent_recovery = IndependentRecovery(client)

    building_damage = Dataset.from_file('Joplin_bldg_dmg_result.csv', "ncsa:buildingDamageVer4")
    household_inventory = Dataset.from_file('hui_ver2-0-0_Joplin_MO_2010_rs1000.csv', "incore:housingUnitInventory")
    household_allocation = Dataset.from_file('Joplin_hua_result_1238.csv', "incore:housingUnitAllocation")
    household_dislocation = Dataset.from_file('pop-dislocation-results.csv', "incore:householdDislocation")
    residential_recovery = Dataset.from_file('joplin_recovery_recovery.csv', "incore:buildingRecovery")
    housing_recovery = Dataset.from_file('housing_recovery_result.csv', "incore:housingRecovery")

    independent_recovery.set_input_dataset("building damage", building_damage)
    independent_recovery.set_input_dataset("household inventory", household_inventory)
    independent_recovery.set_input_dataset("household allocation", household_allocation)
    independent_recovery.set_input_dataset("household dislocation", household_dislocation)
    independent_recovery.set_input_dataset("residential recovery", residential_recovery)
    independent_recovery.set_input_dataset("household housing recovery", housing_recovery)

    # Specify the result name
    result_name = "independent_recovery"

    # Set analysis parameters
    independent_recovery.set_parameter("result_name", result_name)

    # Run the analysis (NOTE: with SettingWithCopyWarning)
    independent_recovery.run_analysis()


if __name__ == '__main__':
    run_with_base_class()
