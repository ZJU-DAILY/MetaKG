# MetaKG

This is our Pytorch implementation for the paper:

> Yuntao Du, Xinjun Zhu, Lu Chen, Ziquan Fang and Yunjun Gao (2022). MetaKG: Meta-learning on Knowledge Graph for Cold-start Recommendation. Paper in [IEEE Explore](https://ieeexplore.ieee.org/document/9760046) or Paper in [arXiv](https://arxiv.org/abs/2204.04959). In IEEE Transactions on Knowledge and Data Engineering, TKDE’ 22.

## Citation

If you want to use our codes and datasets in your research, please cite:

```
@article{MetaKG22,
  author    = {Yuntao Du and
               Xinjun Zhu and
               Lu Chen and
               Ziquan Fang and 
               Yunjun Gao},
  title     = {{MetaKG:}  Meta-learning on Knowledge Graph for Cold-start Recommendation},
  journal   = {{TKDE}},
  year      = {2022}
}
```

## Environment Requirements

- Ubuntu OS
- Python >= 3.8 (Anaconda3 is recommended)
- PyTorch 1.7+
- A Nvidia GPU with cuda 11.1+

## Datasets

We user three popular datasets Amazon-book, Last-FM, and Yelp2018 to conduct experiments.
* We follow the paper "[KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854)" to process data.
* In order to construct cold-start scenario, we find user registration time, item publication time or first interaction time in the full version of recommendation datasets. Then we divide new and old ones in chronological order. 
* For Amazon-book, download book reviews (5-core) from [here](http://jmcauley.ucsd.edu/data/amazon), put it into related `rawdata` folder.
* For Last-FM, download LFM-1b dataset from [here](http://www.cp.jku.at/datasets/LFM-1b/), unzip and put it into related `rawdata` folder.
* For Yelp2018, download Yelp2018 version dataset from [here](https://www.heywhale.com/mw/dataset/5ecbc342fac16e0036ec41a0),  unzip and put it into related `rawdata` folder.

The prepared folder structure is like this:

```
- Datasets
    - pretrain
    - amazon-book
	- rawdata
		- reviews_Books_5.json.gz
    - last-fm
    	- rawdata
    		- LFM-1b_albums.txt
    		- LFM-1b_artists.txt
    		- ...
    - yelp2018
    	- rawdata
    		- yelp_academic_dataset_business.json
    		- yelp_academic_dataset_checkin.json
    		- ...
```

## Train

1. Now, we have provided the cold-start scenario data of last-fm. The codes for constructing the other datasets is as follows.
   ```shell
   python construct_data.py
   ```

2. Start training
   
   Here, we have provided the "meta-model" after meta-training, so you can adapt directly to cold-start scenarios.
   ```shell
   python main.py --dataset last-fm --use_meta_model True
   ```
   You can also retrain the entire model.
   ```shell
   python main.py --dataset last-fm --use_meta_model False
   ```

## Reference

* You can find other baselines in Github.

## Acknowledgement
Any scientific publications that use our datasets should cite the following paper as the reference:

```
@article{MetaKG22,
  author    = {Yuntao Du and
               Xinjun Zhu and
               Lu Chen and
               Ziquan Fang and 
               Yunjun Gao},
  title     = {{MetaKG:}  Meta-learning on Knowledge Graph for Cold-start Recommendation},
  journal   = {{TKDE}},
  year      = {2022}
}
```

Nobody guarantees the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:

* The user must acknowledge the use of the data set in publications resulting from the use of the data set.
* The user may not redistribute the data without separate permission.
* The user may not try to deanonymise the data.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from us.
