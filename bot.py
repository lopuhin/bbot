import interface as bbox


def get_action_by_state(state, verbose=0):
    if verbose:
        for i in xrange(n_features):
            print "state[%d] = %f" %  (i, state[i])

        print "score = {}, time = {}".format(bbox.get_score(), bbox.get_time())

    action_to_do = 0
    return action_to_do


n_features = n_actions = -1


def prepare_bbox():
    global n_features, n_actions

    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()


def run_bbox(verbose=False):
    has_next = 1

    prepare_bbox()

    while has_next:
        state = bbox.get_state()
        action = get_action_by_state(state)
        has_next = bbox.do_action(action)

    bbox.finish(verbose=1)


if __name__ == "__main__":
    run_bbox(verbose=0)
