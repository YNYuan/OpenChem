# Code referenced from
# https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.misc


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
        #                                              simple_value=value)])
        # self.writer.add_summary(summary, step)
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        his_min = float(np.min(values))
        his_max = float(np.max(values))
        his_num = int(np.prod(values.shape))
        his_sum = float(np.sum(values))
        his_sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        bucket_limit = []
        for edge in bin_edges:
            bucket_limit.append(edge)
        bucket = []
        for c in counts:
            bucket.append(c)

        # Create and write Summary
        self.writer.add_histogram_raw(tag, his_min, his_max, his_num, his_sum, his_sum_squares,
                          bucket_limit, bucket, step)
        self.writer.flush()

    def close():
        self.writer.close()
