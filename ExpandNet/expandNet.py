import subprocess
import sys

def run_command(command):
    print(f"\n>>> Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {command}")
        sys.exit(result.returncode)

def main():
    print("Starting ExpandNet script execution...\n")

    # Step 1: Run translation - already done
    # run_command("python3 ExpandNet\\run_translation.py")

    # Step 2: Build dictionary
    run_command("python3 ExpandNet\\build_dict.py")

    # Step 3: Run run_align.py
    run_command(
        "python3 ExpandNet\\run_align.py "
        "--translation_df_file ExpandNet\\expandnet_step1_translate_ur.out.tsv "
        "--lang_src en "
        "--lang_tgt ur "
        "--aligner dbalign "
        "--dict ExpandNet\\dictionaries\\en_ur_dict.tsv "
        "--output_file ExpandNet\\expandnet_step2_align_ur.out.tsv"
    )

    # Step 4: Run run_projection.py
    run_command(
        "python3 ExpandNet\\run_projection.py "
        "--src_data ExpandNet\\Data\\xlwsd_se13.xml "
        "--src_gold ExpandNet\\Data\\se13.key.txt "
        "--dictionary ExpandNet\\dictionaries\\en_ur_dict.tsv "
        "--alignment_file ExpandNet\\expandnet_step2_align_ur.out.tsv "
        "--output_file ExpandNet\\expandnet_step3_project_ur.out.tsv "
        "--join_char _"
    )

    print("\nExpandNet scripts finished.")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
