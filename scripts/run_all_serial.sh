#!/bin/bash
# Run each variant in isolation. Resume-aware via existing skip logic.
# Works around the cross-variant bug in run_squid_batch.py multi-variant mode.
 
cd "$(dirname "$0")/.."
 
VARIANTS=(
    bird_type1_contradictory_L1
    bird_type1_contradictory_L2
    bird_type1_contradictory_L3
    bird_type2_missing_L1
    bird_type2_missing_L2
    bird_type2_missing_L3
    bird_type3_duplicates_L1
    bird_type3_duplicates_L2
    bird_type3_duplicates_L3
    bird_type4_structural_L1
    bird_type4_structural_L2
    bird_type4_structural_L3
)
 
for v in "${VARIANTS[@]}"; do
    echo ""
    echo "=================================================="
    echo "VARIANT: $v"
    echo "Time: $(date)"
    echo "=================================================="
    python scripts/run_squid_batch.py --only "$v"
    echo "--- Completed $v at $(date) ---"
done
 
echo ""
echo "All 12 variants processed."