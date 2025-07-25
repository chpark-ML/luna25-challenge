import json

import yaml


def main():
    # Path to the original YAML file
    model_name = "5_0_9rc8"
    input_yaml_path = f"configs/models/{model_name}.yaml"
    weights_json_path = f"outputs/model_weights/{model_name}/model_weights.json"
    output_yaml_path = f"configs/models/{model_name}_weighted.yaml"

    # Load JSON
    with open(weights_json_path, "r") as f:
        weights_data = json.load(f)

    # Extract model_x keys from model_order and create mapping
    order_to_weight = {
        f"model_{int(name.split('_')[-1])}": weight
        for name, weight in zip(weights_data["model_order"], weights_data["weights"])
    }

    # Load YAML
    with open(input_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Add weight to each model
    for model_key in config:
        if model_key in order_to_weight:
            config[model_key]["weight"] = order_to_weight[model_key]
        else:
            print(f"Warning: No weight found for {model_key}")

    # Save the result
    with open(output_yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Updated YAML saved to: {output_yaml_path}")


if __name__ == "__main__":
    main()
