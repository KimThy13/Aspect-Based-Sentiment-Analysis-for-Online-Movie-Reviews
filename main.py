# main.py
import argparse

def run_phase1_train(config):
    from src.phase1_component_extraction.train import train_phase1
    train_phase1(config)

def run_phase1_infer(config):
    from src.phase1_component_extraction.inference import inference_phase1
    inference_phase1(config)

def run_phase2_train(config):
    from src.phase2_aspect_extraction.train import train_phase2
    train_phase2(config)

def run_phase2_infer(config):
    from src.phase2_aspect_extraction.inference import inference_phase2
    inference_phase2(config)

def main():
    parser = argparse.ArgumentParser()

    # --- chọn phase và action ---
    parser.add_argument("--phase", choices=["phase1", "phase2"], required=True, help="Chọn phase bạn muốn chạy.")
    parser.add_argument("--action", choices=["train", "inference", "both"], required=True, help="Chạy train, inference hoặc cả hai.")

    # --- tham số huấn luyện cơ bản ---
    parser.add_argument("--model_name", type=str, default="t5-small", help="Tên pretrained model")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)

    # --- path dữ liệu ---
    parser.add_argument("--data_dir", type=str, default="data/split")
    parser.add_argument("--output_dir", type=str, default="outputs/")

    args = parser.parse_args()

    # convert args to dict để truyền xuống module khác dễ dàng
    config = vars(args)

    # --- chạy theo phase & action ---
    if args.phase == "phase1":
        if args.action == "train":
            run_phase1_train(config)
        elif args.action == "inference":
            run_phase1_infer(config)
        elif args.action == "both":
            run_phase1_train(config)
            run_phase1_infer(config)

    elif args.phase == "phase2":
        if args.action == "train":
            run_phase2_train(config)
        elif args.action == "inference":
            run_phase2_infer(config)
        elif args.action == "both":
            run_phase2_train(config)
            run_phase2_infer(config)

if __name__ == "__main__":
    main()
