from dataset.unity_dataset import unityDataset
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from model.fplocnet import FpLocNet, quick_fplocnet_call
from eval_utils import *
import os
import pickle
def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)

folder_name="results_norefine_noshuffle_try_wo_scheduler"

if __name__ == "__main__":

    cfg = {
        "Q": 100,
        "Q_refine": 20,
        "D": 128,
        "G": 32,
        "H": 32,
        "dist_max": 10,
        "Vr": 64,
        "V": 16,
        "disable_semantics": False,
        "disable_pointnet": False,
        "fov": 360 ,
        "view_type": "eview",
    }
    model = FpLocNet(cfg).cuda()
    model.load_state_dict(torch.load("try_wo_scheduler.pth"))

    eval_dataset=unityDataset(dataset_dir="unitydataset",is_training=False,n_sample_points=2048,)
    eval_dataloader=DataLoader(dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )


    save_dir = os.path.join("eval_unitydataset", folder_name)
    mkdir_if_not_exist(os.path.join(save_dir, "score_maps"))
    mkdir_if_not_exist(os.path.join(save_dir, "rot_maps"))
    mkdir_if_not_exist(os.path.join(save_dir, "results"))
    mkdir_if_not_exist(os.path.join(save_dir, "query_images"))
    mkdir_if_not_exist(os.path.join(save_dir, "terrs"))
    mkdir_if_not_exist(os.path.join(save_dir, "rerrs"))
    mkdir_if_not_exist(os.path.join(save_dir, "raws"))

    idx=0
    for eval_data in eval_dataloader:
        for k in eval_data.keys():
            if torch.is_tensor(eval_data[k]) and not eval_data[k].is_cuda:
                eval_data[k] = eval_data[k].cuda()

        if cfg["disable_semantics"]:
            eval_data["bases_feat"][..., -2:] = 0
        sample_ret = sample_floorplan(
            eval_data,
            model,
            cfg,
            sample_grid=0.1,
            batch_size=256,
        )
        match_ret = match_images(
            sample_ret,
            eval_data,
            model,
            cfg,
            mode="match",
            sample_nrots=16,
            max_refine_its=3,
        )
        fmt = {
            "idx": idx,
            "sampling_fps": sample_ret["sampling_fps"],
            "sampling_time": sample_ret["sampling_time"],
            "matching_fps": match_ret["matching_fps"],
            "matching_time": match_ret["matching_time"],
            "median_terr": np.median(match_ret["terrs"]),
            "median_rerr": np.median(match_ret["rerrs"]),
        }
        print(
            "{idx}, {sampling_fps:.0f} sampling_fps, {sampling_time:.2f} sampling_time, {matching_fps:.2f} matching_fps, {matching_time:.2f} matching_time, {median_terr:.4f} median_terr, {median_rerr:.4f} median_rerr".format(
                **fmt
            )
        )

        for i in range(match_ret["n_images"]):
            score_map = match_ret["score_maps"][i]
            rot_map = match_ret["rot_maps"][i]
            loc_gt = match_ret["loc_gts"][i]
            loc_est = match_ret["loc_ests"][i]

            score_map_viz = (
                cv2.resize(score_map, (0, 0), fx=2, fy=2) ** 2 * 255
            ).astype(np.uint8)
            rot_map_viz = (
                cv2.resize(rot_map / (2 * np.pi), (0, 0), fx=2, fy=2) * 255
            ).astype(np.uint8)
            result_viz = render_result(
                eval_data["bases"][0].cpu().numpy(),
                eval_data["bases_feat"][0].cpu().numpy(),
                loc_gt,
                loc_est,
            )
            query_image_viz = eval_data["query_image"][i].permute(1, 2, 0).cpu().numpy() * (
                0.229,
                0.224,
                0.225,
            ) + (0.485, 0.456, 0.406)
            query_image_viz = cv2.cvtColor(
                (query_image_viz * 255).astype(np.uint8), cv2.COLOR_BGR2RGB
            )

            if i == 0:
                np.savetxt(
                    os.path.join(save_dir, "terrs", f"terrs_{idx:04d}.txt"),
                    match_ret["terrs"],
                )
                np.savetxt(
                    os.path.join(save_dir, "rerrs", f"rerrs_{idx:04d}.txt"),
                    match_ret["rerrs"],
                )
                with open(
                    os.path.join(save_dir, "raws", f"raw_{idx:04d}.pkl"), "wb"
                ) as f:
                    merged_raw = {**sample_ret, **match_ret, **eval_data}
                    merged_raw["idx"] = idx
                    merged_raw[
                        "samples_feat"
                    ] = None  # too large to store, render in runtime
                    merged_raw["fp_feat"] = None
                    merged_raw["img_feats"] = None
                    for k in merged_raw:
                        if torch.is_tensor(merged_raw[k]):
                            merged_raw[k] = merged_raw[k].cpu().numpy()
                    pickle.dump(merged_raw, f)

            cv2.imwrite(
                os.path.join(
                    save_dir, "score_maps", f"score_map_{idx:04d}_{i:03d}.png"
                ),
                score_map_viz,
            )
            cv2.imwrite(
                os.path.join(
                    save_dir, "rot_maps", f"rot_map_{idx:04d}_{i:03d}.png"
                ),
                rot_map_viz,
            )
            cv2.imwrite(
                os.path.join(save_dir, "results", f"result_{idx:04d}_{i:03d}.png"),
                result_viz,
            )
            cv2.imwrite(
                os.path.join(
                    save_dir, "query_images", f"query_image_{idx:04d}_{i:03d}.png"
                ),
                query_image_viz,
            )
        idx+=1

