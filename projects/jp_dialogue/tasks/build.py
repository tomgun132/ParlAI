import os
import parlai.core.params as params
import parlai.core.build_data as build_data


URL_ROOT = 'https://drive.google.com/open?id=1-XZpZ-K_zTVAukZrQqEoijPe87Ujc7Ic'
FOLDER_NAME = 'jp_dialogue'


def build(opt):
    dpath = os.path.join(opt['datapath'], opt.get('train_folder', FOLDER_NAME))
    # version 1.0: initial release
    # version 1.1: add evaluation logs
    version = '1.1'

    if not build_data.built(dpath, version_string=version):
        if build_data.built(dpath):
            # older version exists, so remove the outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # first download the twitter data files
        # fname_data = 'twitter_data.tar.gz'
        # build_data.download(URL_ROOT, dpath, fname_data)
        # build_data.untar(dpath, fname_data)

        # next download the real dialogue data
        # fname_evallogs = 'evaluationlogs_v1.tar.gz'
        # build_data.download(URL_ROOT + fname_evallogs, dpath, fname_evallogs)
        # build_data.untar(dpath, fname_evallogs)

        print("Data has been placed in " + dpath)

        build_data.mark_done(dpath, version)


def make_path(opt, fname):
    return os.path.join(opt['datapath'], opt.get('train_folder', FOLDER_NAME), fname)


if __name__ == '__main__':
    opt = params.ParlaiParser().parse_args(print_args=False)
    build(opt)
