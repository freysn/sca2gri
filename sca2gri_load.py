import helper_sca2gri as helper

def load_emb_rep(emb_fname, rep_fname):

    embedding=helper.pkl_load(emb_fname)

    if embedding is None:
        print(f'embedding could not be loaded from {emb_fname}')
        sys.exit(0)

    if rep_fname.endswith('.pkl'):
        rep=helper.pkl_load(rep_fname)

        if rep is None:
            print(f'rep could not be loaded from {rep_fname}')
            sys.exit(0)
    else:
        import rep_imgs
        rep=rep_imgs.rep_imgs(rep_fname)

        if len(rep)==0:
            print(f'rep could not be loaded from {rep_fname}')
            sys.exit(0)

    return embedding, rep
