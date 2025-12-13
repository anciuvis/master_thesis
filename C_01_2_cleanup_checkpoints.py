# ========================================================================
# CHECKPOINT CLEANUP SCRIPT
# Use this to clean old checkpoints before re-running clustering
# ========================================================================

import os
import sys
from pathlib import Path
from datetime import datetime

CHECKPOINT_DIR = Path('C:/Users/Anya/master_thesis/output/checkpoints_hierarch')

def print_header(text):
    print("\n" + "="*80)
    print(text)
    print("="*80)

def print_file_list(files, title):
    if not files:
        print(f"\n{title}: (none)")
        return
    
    print(f"\n{title}:")
    for f in sorted(files):
        print(f"  - {f.name}")

def clean_checkpoints(keep_data=False, dry_run=False):
    """
    Clean old checkpoints
    
    Args:
        keep_data: If True, keeps df_sample (reuses data)
                   If False, deletes everything
        dry_run: If True, shows what would be deleted without deleting
    """
    
    if not CHECKPOINT_DIR.exists():
        print_header("ERROR: Checkpoint Directory Not Found")
        print(f"Path: {CHECKPOINT_DIR}")
        print("\nCreate the directory or check the path")
        return False
    
    print_header("CHECKPOINT CLEANUP")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Keep data sample: {keep_data}")
    print(f"Dry run (no deletion): {dry_run}")
    
    # Get all files
    all_files = list(CHECKPOINT_DIR.glob('*'))
    
    if not all_files:
        print("\nNo files found in checkpoint directory")
        return True
    
    # Categorize files
    to_delete = []
    to_keep = []
    
    for file in all_files:
        filename = file.name
        
        # Decide what to delete
        if keep_data and 'df_sample' in filename:
            to_keep.append(file)
        else:
            to_delete.append(file)
    
    # Print summary
    print_file_list(to_delete, "Files to DELETE")
    print_file_list(to_keep, "Files to KEEP")
    
    print(f"\n" + "-"*80)
    print(f"Summary:")
    print(f"  Total files: {len(all_files)}")
    print(f"  To delete: {len(to_delete)}")
    print(f"  To keep: {len(to_keep)}")
    
    # Delete files
    if not dry_run:
        if to_delete:
            confirm = input(f"\nDelete {len(to_delete)} files? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                for file in to_delete:
                    try:
                        file.unlink()
                        print(f"  [DELETED] {file.name}")
                    except Exception as e:
                        print(f"  [ERROR] Failed to delete {file.name}: {e}")
                        return False
                
                print(f"\nSuccessfully deleted {len(to_delete)} files")
                print("Ready to re-run clustering script")
                return True
            else:
                print("Cancelled")
                return False
        else:
            print("\nNo files to delete")
            return True
    else:
        print("\n[DRY RUN] No files were actually deleted")
        print("Run without --dry-run to actually delete files")
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Clean checkpoint files before re-running clustering'
    )
    parser.add_argument(
        '--keep-data',
        action='store_true',
        help='Keep df_sample checkpoint (reuse data, faster re-run)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Delete without asking for confirmation'
    )
    
    args = parser.parse_args()
    
    print("\n")
    success = clean_checkpoints(keep_data=args.keep_data, dry_run=args.dry_run)
    
    if not success:
        sys.exit(1)
    
    print("\n" + "="*80)
    print("NEXT STEP: Run clustering script")
    print("="*80)
    print("\nCommand:")
    print("  python hierarchical_clustering_finegrained.py train_6")
    print("\nExpected result:")
    print("  - ~80 spatial zones created")
    print("  - Better geographic separation than before")
    print("  - Improved visualization")

if __name__ == "__main__":
    main()