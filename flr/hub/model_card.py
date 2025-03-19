class ModelCard:
    def __init__(self, model_name,
                       flops,
                       cost_per_token,
                       params,
                       landmarks,
                       **kwargs):
        
        """
        Args:
            model_name: Name of the model
            flops: Number of FLOPS
            cost_per_token: Cost per token
            params: Number of parameters
            landmarks: List of landmarks
        
        Returns:
            None
        """

        self.model_name = model_name
        self.flops = flops
        self.cost_per_token = cost_per_token
        self.params = params
        self.landmarks = landmarks


    def get_info(self):
        
        return {
            "model_name": self.model_name,
            "flops": self.flops,
            "cost_per_token": self.cost_per_token,
            "params": self.params,
            "landmarks": self.landmarks
            }
    
    def load_model_card_from_file(self):
        pass