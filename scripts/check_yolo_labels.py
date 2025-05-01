# 수정된 라벨 검증 스크립트
from collections import Counter
import os
import yaml
from pathlib import Path

def verify_labels(yaml_path, label_dir):
    # YAML 클래스 로드
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
        yaml_classes = data['names']
    
    # 라벨 분석
    class_counts = Counter()
    for file in Path(label_dir).glob("*.txt"):
        with open(file) as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    
    # 매핑 확인
    print("YAML 클래스 vs 라벨 클래스:")
    for class_id, count in class_counts.items():
        class_name = yaml_classes[class_id] if class_id < len(yaml_classes) else "Unknown"
        print(f"ID {class_id}: {class_name} → {count} instances")

verify_labels("data/rellis3d/rellis3d.yaml", "data/rellis3d/labels/train")
