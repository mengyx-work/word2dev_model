import stat, os


def create_local_model_path(common_path, model_name):
    return os.path.join(common_path, model_name)


def create_local_log_path(common_path, model_name):
    return os.path.join(common_path, model_name, "log")


def generate_tensorboard_script(logdir):
    file_name = "start_tensorboard.sh"
    with open(file_name, "w") as text_file:
        text_file.write("#!/bin/bash \n")
        text_file.write("tensorboard --logdir={}".format(logdir))
    st = os.stat(file_name)
    os.chmod(file_name, st.st_mode | stat.S_IEXEC)


def model_meta_file(model_path, file_prefix="models"):
    meta_files = [f for f in os.listdir(model_path) if f[-5:] == '.meta']
    final_model_files = [f for f in meta_files if file_prefix in f]
    if len(final_model_files) == 0:
        raise ValueError("failed to find any model meta files in {}".format(model_path))
    if len(final_model_files) > 1:
        print "warning, more than one model meta file is found in {}".format(model_path)
    return os.path.join(model_path, final_model_files[0])


def clear_folder(absolute_folder_path):
    if not os.path.exists(absolute_folder_path):
        os.makedirs(absolute_folder_path)
        return
    for file_name in os.listdir(absolute_folder_path):
        file_path = os.path.join(absolute_folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print 'failed to clear folder {}, with error {}'.foramt(absolute_folder_path, e)
