from thermodrive.thermosyphon import ThermosyphonDesign, max_heat_per_pipe_W, pipe_conductance_W_K


def test_high_output_package_has_higher_capacity_than_economy():
    economy = ThermosyphonDesign(spacing_m=1.2, depth_m=3.0, diameter_m=0.025, fluid="Methanol blend (factory sealed)")
    high = ThermosyphonDesign(spacing_m=1.2, depth_m=3.0, diameter_m=0.038, fluid="High-output CO2 + thermal grout + heat spreader")
    assert max_heat_per_pipe_W(high) > max_heat_per_pipe_W(economy)
    assert pipe_conductance_W_K(high, soil_k_W_mK=1.3, top_k_W_mK=1.2) > pipe_conductance_W_K(economy, soil_k_W_mK=1.3, top_k_W_mK=1.2)
