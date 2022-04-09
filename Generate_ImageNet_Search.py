import os
import argparse
import random


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--proportion_train', type=float, default='0.1', help='proportion of samples')
parser.add_argument('--proportion_val', type=float, default='0.0278', help='proportion of samples')  #  0.025/0.9 = 0.0278
parser.add_argument('--dataset_dir', type=str, default='./dataset', help='folder to copy to')
parser.add_argument('--src_path', type=str, default='/home/dataset/ImageNet')  
parser.add_argument('--name', type=str, default='imagenet_search', help='Dataset name') 
args = parser.parse_args()



class DatasetGen:
    def __init__(self, args):
        self.train_roots = []

        self.args = args#self.init_parser().parse_args()



    def get_paths(self):
        """
        Get ImageNet folder paths
        :return:
        """
        train_roots = self.train_roots
        # import code
        # code.interact(local=locals())
        for root, dirs, files in os.walk(os.path.join(self.args.src_path, 'train')):
            train_roots.append(root)

    def greate_link(self, src_path, target_path):
        # import code
        # code.interact(local=locals())
        base = os.path.basename(src_path)
        target_name = os.path.join(target_path, base)
        os.symlink(src_path, target_name)

    def generate_folders(self):
        args = self.args
        #num = args.num_classes

        dataset_dir = os.path.join(args.dataset_dir, args.name)
        if os.path.exists(dataset_dir):
            raise FileExistsError('%s exists', dataset_dir)

        self.get_paths()
        dic_train={}
        dic_val={}
        for classes in self.train_roots:
            name=classes.split('/')[-1]
            if name!='train':
                files = os.listdir(classes)
                randoms_train = random.sample(range(len(files)), int(args.proportion_train * len(files)))
                # randoms_val = randoms_train
                # while (set(randoms_train)&set(randoms_val))!=0:
                no_train=[i for i in range(len(files)) if i not in randoms_train]
                randoms_val = random.sample(no_train, int(args.proportion_val * len(files)))
                src_train=[os.path.join(classes,files[i]) for i in randoms_train]
                src_val = [os.path.join(classes,files[i]) for i in randoms_val]
                # import code
                # code.interact(local=locals())
                dic_train[name]=src_train
                dic_val[name] =src_val



        os.mkdir(dataset_dir)
        target_train_dir = os.path.join(dataset_dir, 'train')
        target_val_dir = os.path.join(dataset_dir, 'val')
        os.mkdir(target_train_dir)
        os.mkdir(target_val_dir)
        for key in dic_train:
            target_train=os.path.join(target_train_dir, key)
            os.mkdir(target_train)
            for i in range(len(dic_train[key])):
                self.greate_link(dic_train[key][i], target_train)
                #self.greate_link(src_val[i], target_val_dir)
        for key in dic_val:
            target_val = os.path.join(target_val_dir, key)
            os.mkdir(target_val)
            for i in range(len(dic_val[key])):
                self.greate_link(dic_val[key][i], target_val)

        # for i in range(len(src_train)):
        #     self.greate_link(src_train[i], target_train_dir)
        #     self.greate_link(src_val[i], target_val_dir)


if __name__ == '__main__':
    g = DatasetGen(args)
    g.generate_folders()