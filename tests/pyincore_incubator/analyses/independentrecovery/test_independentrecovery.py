from pyincore import IncoreClient
from pyincore_incubator.analyses.independentrecovery import IndependentRecovery
import pyincore.globals as pyglobals


def run_with_base_class():
    # Connect to IN-CORE service
    client = IncoreClient(pyglobals.INCORE_API_DEV_URL)
    client.clear_cache()

    independent_recovery = IndependentRecovery(client)

    building_dmg_id = "6824dff3ce15ce6bdf893282"
    housing_inventory_id = "6824e19fce15ce6bdf89328a"
    housing_unit_id = "6824e277ce15ce6bdf893293"
    population_dislocation_id = "6824e49ace15ce6bdf8932a5"
    residential_recovery_id = "6824e4c1ce15ce6bdf8932a6"
    housing_recovery_id = "6824e4fdce15ce6bdf8932a7"

    independent_recovery.load_remote_input_dataset("building_damage", building_dmg_id)
    independent_recovery.load_remote_input_dataset(
        "housing_unit_inventory", housing_inventory_id
    )

    independent_recovery.load_remote_input_dataset(
        "housing_unit_allocation", housing_unit_id
    )

    independent_recovery.load_remote_input_dataset(
        "population_dislocation", population_dislocation_id
    )

    independent_recovery.load_remote_input_dataset(
        "residential_recovery", residential_recovery_id
    )
    independent_recovery.load_remote_input_dataset(
        "household_housing_recovery", housing_recovery_id
    )

    # Specify the result name
    result_name = "independent_recovery"

    # Set analysis parameters
    independent_recovery.set_parameter("result_name", result_name)

    # Run the analysis (NOTE: with SettingWithCopyWarning)
    independent_recovery.run_analysis()


if __name__ == "__main__":
    run_with_base_class()
