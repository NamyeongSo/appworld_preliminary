# Counterfactual Evaluation Report
Target directory:
`/home/thskadud/appworld/experiments/outputs/counterfactual/google/gemma-4-26b-a4b-it`

## Method
Read all `counterfactual/manifest.json` files under the target directory.

Definitions:
- **Root success**: `root_success == true`
- **Root + counterfactual success**: success if either:
  - `root_success == true`, or
  - `final_success_trajectory_id` is not `null`
- **Trajectory count**: `num_trajectories` from each manifest, which includes the root trajectory

## Sample size
- Total manifests read: **114**

## Results

### 1. Root AppWorld success rate
- **80 / 114**
- **70.18%**

### 2. Root + counterfactual AppWorld success rate
- **108 / 114**
- **94.74%**

### 3. Average number of trajectories per task
(root + counterfactual, including root)
- Total trajectories: **990**
- Average: **8.68**
- Median: **6**
- Min: **2**
- Max: **49**

### 4. Average attempts needed to turn root failure into success
Only for cases where:
- root failed, and
- a counterfactual trajectory eventually succeeded

Results:
- Rescued cases: **28**
- Total additional counterfactual attempts: **107**
- Average additional attempts needed: **3.82**
- Median: **3**
- Min: **1**
- Max: **9**

Reference:
- Average total trajectories for rescued cases, including root: **4.82**

## Failure note
- Final unresolved failures: **6**
- These had:
  - `final_success_trajectory_id = null`
  - `pair_complete = false`

## Summary
Counterfactual recovery substantially improved overall performance:
- from **70.18%** with root only
- to **94.74%** with root + counterfactual

When root initially failed, successful recovery required about **3.82 extra attempts on average**.
