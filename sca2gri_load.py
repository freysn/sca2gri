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



    #
    # normalize embedding
    #
    do_normalize_embedding=True
    if do_normalize_embedding:
        minv=(min(embedding[:, 0]), min(embedding[:, 1]))
        maxv=(max(embedding[:, 0]), max(embedding[:, 1]))

        max_extent=max(maxv[0]-minv[0],maxv[1]-minv[1])

        embedding[:, 0]-=minv[0]
        embedding[:, 1]-=minv[1]

        embedding /= max_extent

    print(f'#elements: embedding {len(embedding)} | rep {len(rep)}')

    assert len(embedding) == len(rep)

    return embedding, rep
