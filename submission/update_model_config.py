import yaml
import json


def main():
    # 원본 YAML 파일 경로
    input_yaml_path = "configs/models/5_0_9.yaml"
    weights_json_path = "model_weights/5_0_9rc8/model_weights.json"
    output_yaml_path = "configs/models/5_0_9.yaml"


    # JSON 로드
    with open(weights_json_path, "r") as f:
        weights_data = json.load(f)

    # model_order에서 model_x 키 추출 후 mapping 생성
    order_to_weight = {
        f"model_{int(name.split('_')[-1])}": weight
        for name, weight in zip(weights_data["model_order"], weights_data["weights"])
    }

    # YAML 로드
    with open(input_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # 각 모델에 weight 추가
    for model_key in config:
        if model_key in order_to_weight:
            config[model_key]["weight"] = order_to_weight[model_key]
        else:
            print(f"Warning: No weight found for {model_key}")

    # 결과 저장
    with open(output_yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Updated YAML saved to: {output_yaml_path}")


if __name__ == "__main__":
    main()
