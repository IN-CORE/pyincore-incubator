from pyincore import IncoreClient
from pyincore_incubator.analyses.cohortbuildingdamage import CohortBuildingDamage
import pyincore.globals as pyglobals


def run_with_base_class():
    client = IncoreClient(pyglobals.INCORE_API_DEV_URL)

    # Building dataset
    bldg_dataset_id = "5a284f0bc7d30d13bc081a28"

    example = CohortBuildingDamage(client)
    example.load_remote_input_dataset("buildings", bldg_dataset_id)
    example.set_parameter("max_value", 0.5)

    result_name = "example_output"
    example.set_parameter("result_name", result_name)

    # Run Analysis
    example.run_analysis()


if __name__ == "__main__":
    run_with_base_class()
