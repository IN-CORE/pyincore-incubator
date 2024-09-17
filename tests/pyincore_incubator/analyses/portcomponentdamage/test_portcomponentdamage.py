# %%
from pyincore import IncoreClient
from pyincore_incubator.analyses.portcomponentdamage import PortComponentDamage


# %%
def run_with_base_class():
    client = IncoreClient()

    # Component dataset
    component_dataset_id = "65ebeb310db8d44edc1f45c8"

    example = PortComponentDamage(client)

    example.load_remote_input_dataset("components", component_dataset_id)

    print(example)

    # result_name = "example_output"
    # example.set_parameter("result_name", result_name)

    # # Run Analysis
    # example.run_analysis()


if __name__ == "__main__":
    run_with_base_class()
    print("ok")
