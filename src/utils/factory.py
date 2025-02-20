from src.models import registry as model_registry
from src.optimizers import registry as optimizer_registry
from src.losses import registry as loss_registry
from src.functions import registry as function_registry
from src.utils.logger import setup_logger

logger=setup_logger()
# logger = logging.getLogger('train_logger')
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)

def get_model(config: dict):
    model_name = config["model"]["name"]
    model_params = config["model"].get("params", {})
    logger.debug(f"Fetching model: {model_name} with params: {model_params}")
    return model_registry[model_name]

def get_optimizer(config: dict, model):
    opt_name = config["optimizer"]["name"]
    opt_params = config["optimizer"].get("params", {})
    logger.debug(f"Fetching optimizer: {opt_name} with params: {opt_params}")
    
    return optimizer_registry[opt_name](model.parameters(), **opt_params)

def get_loss(config: dict):
    loss_name = config["loss"]["name"]
    loss_params = config["loss"].get("params", {})
    logger.debug(f"Fetching loss function: {loss_name} with params: {loss_params}")
    return loss_registry[loss_name](**loss_params)

def get_function(config: dict):
    function_name = config["function"]["name"]
    function_params = config["function"].get("params", {})
    logger.debug(f"Fetching function: {function_name} with params: {function_params}")
    return function_registry[function_name](**function_params)
