#!/usr/bin/env python3
"""
Test script for ElectricPowerNetworkRecovery analysis.
This is a Python script version of the Jupyter notebook for easier testing.
"""

import warnings
from pyincore import (
    IncoreClient,
    FragilityService,
    MappingSet,
)
from pyincore.analyses.epfdamage import EpfDamage
from pyincore.analyses.montecarlofailureprobability import (
    MonteCarloFailureProbability,
)
from pyincore_incubator.analyses.electricpowernetworkrecovery import (
    ElectricPowerNetworkRecovery,
)
import pyincore.globals as pyglobals

warnings.filterwarnings("ignore")


def main():
    """Main function to run the electric power network recovery analysis test."""

    print("Starting Electric Power Network Recovery Analysis Test...")

    # Initialize client and services
    print("Connecting to IN-CORE services...")
    # client = IncoreClient(pyglobals.INCORE_API_DEV_URL)
    client = IncoreClient(pyglobals.INCORE_API_PROD_URL)
    client.clear_cache()
    fragility_service = FragilityService(client)
    print("Connection successful to IN-CORE services.")

    # Define dataset IDs and parameters
    node_dataset_id = "66912740acbe296a1b8145b3"
    edge_dataset_id = "6690763bacbe296a1b8144b3"
    building_dataset_id = "63ff6b135c35c0353d5ed3ac"  # Island Only

    hazard_type = "hurricane"
    # 100yr Hazard Event
    hazard_id = "5fa5a9497e5cdf51ebf1add2"
    # Number of samples
    sample_num = 12
    crew_num = 20

    # Step 1: Run EPF Damage Analysis
    print("Running EPF Damage Analysis...")

    # EPF fragility mapping
    epf_mapping_id = "62fac92ecef2881193f22613"
    epf_mapping_set = MappingSet(fragility_service.get_mapping(epf_mapping_id))

    epf_dmg_hurricane_galveston = EpfDamage(client)
    epf_dmg_hurricane_galveston.load_remote_input_dataset(
        "epfs", "62fc000f88470b319561b58d"
    )
    epf_dmg_hurricane_galveston.set_input_dataset("dfr3_mapping_set", epf_mapping_set)
    epf_dmg_hurricane_galveston.set_parameter(
        "result_name", "Galveston-hurricane-epf-damage"
    )
    epf_dmg_hurricane_galveston.set_parameter(
        "fragility_key", "Non-Retrofit Fragility ID Code"
    )
    epf_dmg_hurricane_galveston.set_parameter("hazard_type", hazard_type)
    epf_dmg_hurricane_galveston.set_parameter("hazard_id", hazard_id)
    epf_dmg_hurricane_galveston.set_parameter("num_cpu", 8)

    # Run Analysis
    epf_dmg_hurricane_galveston.run_analysis()
    epf_dmg_result = epf_dmg_hurricane_galveston.get_output_dataset("result")
    epf_dmg_result_df = epf_dmg_result.get_dataframe_from_csv()
    print(f"EPF Damage Analysis completed. Results shape: {epf_dmg_result_df.shape}")
    print("EPF Damage Results (first 5 rows):")
    print(epf_dmg_result_df.head())

    # Step 2: Run Monte Carlo Failure Probability Analysis
    print("\nRunning Monte Carlo Failure Probability Analysis...")

    mc = MonteCarloFailureProbability(client)
    mc.set_input_dataset("damage", epf_dmg_result)
    mc.set_parameter("result_name", "epf_mc")
    mc.set_parameter("num_cpu", 8)
    mc.set_parameter("num_samples", sample_num)
    mc.set_parameter("damage_interval_keys", ["DS_0", "DS_1", "DS_2", "DS_3"])
    mc.set_parameter("failure_state_keys", ["DS_1", "DS_2", "DS_3"])
    mc.run_analysis()

    gal_pole_damage_mcs_samples = mc.get_output_dataset("sample_failure_state")
    gal_pole_damage_mcs_samples_df = (
        gal_pole_damage_mcs_samples.get_dataframe_from_csv()
    )
    print(
        f"Monte Carlo Analysis completed. Results shape: {gal_pole_damage_mcs_samples_df.shape}"
    )
    print("Monte Carlo Results (first 5 rows):")
    print(gal_pole_damage_mcs_samples_df.head())

    # Step 3: Run Electric Power Network Recovery Analysis
    print("\nRunning Electric Power Network Recovery Analysis...")

    power_recovery_analysis = ElectricPowerNetworkRecovery(client)

    power_recovery_analysis.load_remote_input_dataset("nodes", node_dataset_id)
    power_recovery_analysis.load_remote_input_dataset("edges", edge_dataset_id)
    power_recovery_analysis.load_remote_input_dataset("buildings", building_dataset_id)
    power_recovery_analysis.set_input_dataset(
        "pole_damage", gal_pole_damage_mcs_samples
    )
    power_recovery_analysis.set_parameter("result_name", "power_analysis_output")
    power_recovery_analysis.set_parameter("num_cpu", 8)
    power_recovery_analysis.set_parameter("num_samples", sample_num)
    power_recovery_analysis.set_parameter("num_crews", crew_num)
    power_recovery_analysis.set_parameter("time_step", 7)
    power_recovery_analysis.set_parameter("consider_protective_devices", False)

    # Run Analysis
    power_recovery_analysis.run_analysis()

    # Get outputs
    power_test_result = power_recovery_analysis.get_output_dataset("result")
    power_test_result_df = power_test_result.get_dataframe_from_csv()

    print(
        f"Electric Power Network Recovery Analysis completed. Results shape: {power_test_result_df.shape}"
    )
    print("Power Recovery Results (first 5 rows):")
    print(power_test_result_df.head())

    print("\nAnalysis completed successfully!")

    return power_test_result_df


if __name__ == "__main__":
    try:
        results = main()
        print("\nFinal results summary:")
        print(f"Total buildings analyzed: {len(results)}")
        print(
            f"Average power back time: {results['average_power_back_time'].mean():.2f} days"
        )
        print(
            f"Max power back time: {results['average_power_back_time'].max():.2f} days"
        )
        print(
            f"Min power back time: {results['average_power_back_time'].min():.2f} days"
        )
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        raise
