from pyincore import IncoreClient, Dataset
from pyincore_incubator.analyses.earthquakegenerator import EarthquakeGenerator
import pyincore.globals as pyglobals


def run_with_base_class():
    client = IncoreClient(pyglobals.INCORE_API_DEV_URL)

    magnitude_frequency_distribution = "gr_recurrence_law"
    ground_motion_equation_model = "BooreStewartSeyhanAtkinson2014"

    eq_generator = EarthquakeGenerator(client)
    # bldg_dataset_id = "5a284f0bc7d30d13bc081a28"
    # bridge_dataset_id = "5a284f2ec7d30d13bc08205e"
    # epf_dataset_id = "6189c103d5b02930aa3efc35"
    wf_dataset_id = "5a284f2ac7d30d13bc081e70"
    eq_generator.load_remote_input_dataset("inventory", wf_dataset_id)

    # TODO this should be added to the IN-CORE service
    source_data = Dataset.from_file("source_data.json", "incore:eqData")
    result_name = "example_output"
    eq_generator.set_parameter("result_name", result_name)
    eq_generator.set_parameter(
        "magnitude_frequency_distribution", magnitude_frequency_distribution
    )
    eq_generator.set_parameter(
        "ground_motion_equation_model", ground_motion_equation_model
    )
    eq_generator.set_parameter("ground_motion_type", "SA")
    eq_generator.set_parameter("period", 0.2)
    eq_generator.set_parameter("return_period", 50)
    eq_generator.set_parameter("num_scenarios", 30)
    eq_generator.set_parameter("exceedance_probability", 0.05)
    eq_generator.set_input_dataset("earthquake_data", source_data)

    # Run Analysis
    eq_generator.run_analysis()


if __name__ == "__main__":
    run_with_base_class()
