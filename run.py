from optparse import OptionParser
from dataset import load_ordinary
from kmeans import cluster
from visualize import visualize

def parse_argv():
    parser = OptionParser(description='Naive clustering')
    parser.add_option('-k', '--count', dest='k', metavar='N',
                      help='Cluster count')
    (options, args) = parser.parse_args()
    if len(args) == 0:
        parser.error('at least one dataset file must be specified.')
    if options.k is None:
        parser.error('the count of clusters must be specified.')
    return options, args

if __name__ == '__main__':
    options, args = parse_argv()
    points = load_ordinary(args[0])
    clusters = cluster(points, int(options.k))
    visualize(clusters)
