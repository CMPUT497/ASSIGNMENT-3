import subprocess
import sys
import os

# Full path to your Python interpreter
PYTHON_EXE = r"C:\Users\chiro\AppData\Local\Programs\Python\Python311\python.exe"

def run_command(args):
    print(f"\n>>> Running: {' '.join(args)}")
    result = subprocess.run(args)
    if result.returncode != 0:
        print(f"❌ Command failed: {' '.join(args)}")
        sys.exit(result.returncode)

def main():
    print("Starting ExpandNet script execution...\n")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Step 1: translation - already done
    # run_command([PYTHON_EXE, "run_translation.py"])

    # Step 2: Build dictionary - already done
    # run_command([PYTHON_EXE, "build_dict.py"])

    # Step 3: Run run_align.py - already done
    # run_command([
    #     PYTHON_EXE, "run_align.py",
    #     "--translation_df_file", "expandnet_step1_translate_ur.out.tsv",
    #     "--lang_src", "en",
    #     "--lang_tgt", "ur",
    #     "--aligner", "dbalign",
    #     "--dict", "dictionaries\\en_ur_dict.tsv",
    #     "--output_file", "expandnet_step2_align_ur.out.tsv"
    # ])

    # Step 4: Run run_projection.py
    run_command([
        PYTHON_EXE, "run_projection.py",
        "--src_data", "Data\\xlwsd_se13.xml",
        "--src_gold", "Data\\se13.key.txt",
        "--dictionary", "dictionaries\\en_ur_dict.tsv",
        "--alignment_file", "expandnet_step2_align_ur.out.tsv",
        "--output_file", "expandnet_step3_project_ur.out.tsv",
        "--join_char", "_"
    ])

    print("\n✅ ExpandNet scripts finished successfully.")


if __name__ == "__main__":
    main()
