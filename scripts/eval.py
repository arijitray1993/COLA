# Description: Evaluation metrics for Cola
import json
import numpy as np
import torch
import torchmetrics

def ColaMultiObjAcc(preds):
    # Assumes preds is a 2x2 numpy array like below where the entries are image-caption match scores:
    #         caption 0    caption 1
    # img 0  preds[0, 0]  preds[0, 1]
    # img 1  preds[1, 0]  preds[1, 1]
    
    im0_c0 = preds[0, 0]
    im0_c1 = preds[0, 1]

    im1_c0 = preds[1, 0]
    im1_c1 = preds[1, 1]

    image_correct = im0_c0 > im1_c0 and im1_c1 > im0_c1

    return image_correct

def ColaSingleObjAP(preds, labels, hard_inds):
    # this assumes as input preds which should be num_images x labels image-caption match scores.
    # labels should be num_images x labels binary labels whether the caption label is present in the image
    # hard_inds should be num_images x labels binary labels indicating whether the label is a hard distractor for the image

    # ColaSingleObj data from the github should contain [label_captions, data]. data is a list of list like [image_file, objects_attributes_annotation, label, hard_list]
    # Compute scores for each of the images for each of the labels to get preds. The caption for the labels in the label_captions entry. This should give you num_images x labels image-caption match scores.
    # use the label entry in the ColaSingleObjData to get labels of shape num_images x labels binary labels whether the caption label is present in the image
    # use the hard_list entry in the ColaSingleObjData to get hard_inds of shape num_images x labels binary labels indicating whether the label is a hard distractor for the image. 

    scores = preds
    labels = labels
    im_obj_inds = hard_inds

    scores = torch.cat(scores).T
    labels = torch.cat(labels).T
    im_obj_inds = torch.cat(im_obj_inds).T
    # print(scores.shape, labels.shape)
    # pdb.set_trace()
    obj_level_avg = []
    for s_row, l_row, im_ind in zip(scores, labels, im_obj_inds):
        if torch.sum(l_row) == 0:
            continue
        # in the valtrain set, sometimes, some labels may not have
        # any GT images since the set is a subset of val

        obj_present_inds = torch.where(im_ind == 1)[0]
        if len(obj_present_inds) == 0:
            continue
        score_obj = s_row[obj_present_inds]
        label_obj = l_row[obj_present_inds]
        obj_level_avg.append(
            torchmetrics.functional.average_precision(score_obj, label_obj)
        )
    # pdb.set_trace()
    return torch.mean(torch.stack(obj_level_avg))



if __name__== "__main__":

    # example multi-obj accuracy:
    multi_obj_data = json.load(open("data/COLA_multiobjects_matching_benchmark.json", "r"))

    batch_acc = []
    for image_1, caption_1, image_2, caption_2 in multi_obj_data:
        
        # pred = compute_score_using_your_mmodel(image_1, caption_1, image_2, caption_2)

        dummy_pred = np.array([[0.1, 0.9], [0.9, 0.1]])
        batch_acc.append(ColaMultiObjAcc(dummy_pred))
    
    print("Multi-obj accuracy: ", np.mean(batch_acc))

    # example single-obj AP:
    single_obj_data = json.load(open("data/COLA_singleobjects_benchmark_GQA.json", "r"))
    
    caption_labels = single_obj_data[0]
    # should be a list like ["square white plate", ....]
    # your prediction should have a 1 when the label is present. So it should also be a list of the same size as caption_labels eg [1,0,....]

    data = single_obj_data[1]

    dummy_preds = []
    for image_file, objects_attributes_annotation, label, hard_list in data:
        # pred = compute_score_using_your_mmodel(image_file, caption_labels)
        dummy_pred = np.random.rand(len(caption_labels))
        dummy_preds.append(dummy_pred)
    
    dummy_labels = torch.tensor([label for image_file, objects_attributes_annotation, label, hard_list in data])
    dummy_hard_inds = torch.tensor([hard_list for image_file, objects_attributes_annotation, label, hard_list in data])

    print("Single-obj AP: ", ColaSingleObjAP(dummy_preds, dummy_labels, dummy_hard_inds))

