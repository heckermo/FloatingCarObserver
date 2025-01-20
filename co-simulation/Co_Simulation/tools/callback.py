
class CoSimulationCallback:
    """
    Base class for simulation callbacks.
    """
    def on_simulation_start(self, simulation):
        """Called at the start of the simulation."""
        pass

    def on_simulation_step(self, simulation, step, sensor_data):
        """Called at each step of the simulation."""
        pass

    def on_simulation_end(self, simulation):
        """Called at the end of the simulation."""
        pass

