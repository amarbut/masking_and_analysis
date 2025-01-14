import os
import argparse
import re


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', help = 'location of model checkpoints', required = True)
    parser.add_argument('--save_steps', help = 'number of steps per checkpoint', type = int, required = True)
    parser.add_argument('--num_chk', help = 'number of checkpoints saved per epoch', type = float, required = True)
    parser.add_argument('--final_save', help = 'checkpoint of final model', type = int, required = True)
    

kwargs = vars(parser.parse_args())


# chk_list = ["checkpoint-"+ str(i) for i in list(range(311, 18666, 311))]
# e_name_list = []
# for chk in chk_list:
for chk in os.listdir(kwargs["model_folder"]):
    print(chk)
    if re.search(r"checkpoint-(\d+)", chk):
        step = int(re.search(r"checkpoint-(\d+)", chk).group(1))
        if step == kwargs["final_save"]:
            e_name = "final"
        else:
            chk_num = step/kwargs["save_steps"]
            epoch = round(chk_num/kwargs["num_chk"], 1)
            e_name = "epoch_"+str(epoch).replace(".", "_")
        print(e_name)
        os.rename(kwargs["model_folder"]+"/"+chk, kwargs["model_folder"]+"/"+e_name)
        #e_name_list.append(e_name)
