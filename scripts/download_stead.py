#!/usr/bin/env python3
"""STEAD 데이터셋 다운로드 헬퍼.

STEAD (STanford EArthquake Dataset)를 SeisBench를 통해 다운로드하거나
직접 다운로드 안내.

사용법:
    # SeisBench를 통한 다운로드 (권장)
    python -m scripts.download_stead --method seisbench --output-dir ./data/stead

    # 직접 다운로드 안내
    python -m scripts.download_stead --method manual
"""

import argparse
import sys
from pathlib import Path


def download_via_seisbench(output_dir):
    """SeisBench를 통해 STEAD 다운로드."""
    try:
        import seisbench.data as sbd
    except ImportError:
        print("SeisBench가 설치되지 않았습니다.")
        print("pip install seisbench 로 설치해주세요.")
        sys.exit(1)

    print("SeisBench를 통해 STEAD 데이터셋을 다운로드합니다...")
    print("(첫 다운로드 시 수십 GB, 시간이 걸릴 수 있습니다)")
    print()

    data = sbd.STEAD(force=False)
    print(f"STEAD 다운로드 완료!")
    print(f"총 샘플 수: {len(data)}")
    print(f"캐시 위치: {sbd.STEAD.path}")
    print()
    print("SeisBench 형식으로 저장되어 있습니다.")
    print("직접 학습에 사용하려면 HDF5/CSV 형식이 필요합니다.")
    print("아래 URL에서 원본 형식을 다운로드하세요:")
    print_manual_instructions()


def print_manual_instructions():
    """직접 다운로드 안내."""
    print()
    print("=" * 60)
    print("STEAD 데이터셋 직접 다운로드 안내")
    print("=" * 60)
    print()
    print("1. GitHub 저장소:")
    print("   https://github.com/smousavi05/STEAD")
    print()
    print("2. 데이터 파일 (약 20GB):")
    print("   - merged.hdf5: 파형 데이터 (1,265,657 traces)")
    print("   - merged.csv:  메타데이터 (P/S arrival 정보 포함)")
    print()
    print("3. 다운로드 방법:")
    print("   a) STEAD GitHub README의 다운로드 링크 사용")
    print("   b) 또는 아래 DOI로 접근:")
    print("      https://doi.org/10.1109/ACCESS.2019.2947848")
    print()
    print("4. 다운로드 후 학습 실행:")
    print("   python -m scripts.train \\")
    print("       --csv /path/to/merged.csv \\")
    print("       --hdf5 /path/to/merged.hdf5 \\")
    print("       --config config/default.yaml")
    print()
    print("5. 소량 테스트 (권장):")
    print("   python -m scripts.train \\")
    print("       --csv /path/to/merged.csv \\")
    print("       --hdf5 /path/to/merged.hdf5 \\")
    print("       --max-samples 1000 --epochs 5")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="STEAD 데이터셋 다운로드")
    parser.add_argument("--method", choices=["seisbench", "manual"],
                        default="manual",
                        help="다운로드 방법 (기본: manual)")
    parser.add_argument("--output-dir", type=str, default="./data/stead",
                        help="출력 디렉토리")
    args = parser.parse_args()

    if args.method == "seisbench":
        download_via_seisbench(args.output_dir)
    else:
        print_manual_instructions()


if __name__ == "__main__":
    main()
