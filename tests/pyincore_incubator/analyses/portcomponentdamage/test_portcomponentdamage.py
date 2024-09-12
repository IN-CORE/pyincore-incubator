from pyincore import IncoreClient
from pyincore_incubator.analyses.portcomponentdamage import PortComponentDamage
import pyincore.globals as pyglobals


def run_with_base_class():
    client = IncoreClient(pyglobals.INCORE_API_DEV_URL, offline=True)

    # Building dataset
    bldg_dataset_id = "5a284f0bc7d30d13bc081a28"

    example = PortComponentDamage(client)
    example.load_remote_input_dataset("buildings", bldg_dataset_id)

    result_name = "example_output"
    example.set_parameter("result_name", result_name)

    # Run Analysis
    example.run_analysis()


if __name__ == "__main__":
    run_with_base_class()
