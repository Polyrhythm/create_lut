import nuke
import os


def write_data(data, dataLabel):
    # calculate colormatrix to match src macbeth chart to dst macbeth chart
    node = nuke.thisNode()

    # make sure numpy is importable
    if node['use_system_python'].getValue():
        import sys
        sys.path.insert(0, '/lib64/python2.7/site-packages')
    try:
        import numpy as np
    except ImportError:
        nuke.message('numpy required. make sure it is installed correctly and importable.')
        return

    FILEPATH = '/home/ryan/Documents/python/data.txt'
    with open(FILEPATH, "a") as f:
        f.write('\n\n{} = np.array([\n'.format(dataLabel))

        for x in data:
            f.write('    [{x}, {y}, {z}],\n'.format(x=x[0], y=x[1], z=x[2]))

        f.write('])\n')


def colorsample(node, pos, size):
    # sample rgb pixel value
    # :param: node - node object to sample
    # :param: pos - list containing x and y position as float values
    # :param: size - box size to average in sample
    return [node.sample(chan, pos[0], pos[1], size, size) for chan in ['red', 'green', 'blue']]


def calc_mtx():
    # calculate colormatrix to match src macbeth chart to dst macbeth chart
    node = nuke.thisNode()

    # make sure numpy is importable
    if node['use_system_python'].getValue():
        import sys
        sys.path.insert(0, '/lib64/python2.7/site-packages')
    try:
        import numpy as np
    except ImportError:
        nuke.message('numpy required. make sure it is installed correctly and importable.')
        return

    size = node['sample_size'].getValue()
    chroma_only = node['chroma_only'].getValue()

    node.begin()
    src = nuke.toNode('Normalize')
    dst = nuke.toNode('macbeth_points')
    norm_node = nuke.toNode('Normalize')

    node['reset'].execute()

    # If chroma_only, normalize colorchecker to dst grey before sampling
    if chroma_only:
        src_grey = colorsample(src, dst['p44'].getValue(), size)
        dst_grey = colorsample(dst, dst['p44'].getValue(), size)

        # with rec709 luminance weighting
        src_lum = (src_grey[0] * 0.2126 + src_grey[1] * 0.7152 + src_grey[2] * 0.0722)
        dst_lum = (dst_grey[0] * 0.2126 + dst_grey[1] * 0.7152 + dst_grey[2] * 0.0722)
        norm_node['src'].setValue(src_lum)
        norm_node['dst'].setValue(dst_lum)

        print
        'source grey: {0} | {1}\ndst grey: {2} | lum: {3}'.format(src_grey, src_lum,
                                                                  dst_grey, dst_lum)

    src_patches = list()
    dst_patches = list()
    points = ['p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p21', 'p22', 'p23', 'p24', 'p25',
              'p26', 'p31', 'p32', 'p33', 'p34', 'p35', 'p36', 'p41', 'p42', 'p43', 'p44', 'p45', 'p46']

    for point in points:
        src_value = colorsample(src, dst[point].getValue(), size)
        dst_value = colorsample(dst, dst[point].getValue(), size)

        src_patches.append(src_value)
        dst_patches.append(dst_value)

        print
        'source: {0}'.format(src_value)
        print
        'destination: {0}'.format(dst_value)

    # Calculate multivariate Rinear Regression to fit source matrix to target matrix
    # https://en.wikipedia.org/wiki/General_linear_model
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
    #
    # https://github.com/colour-science/colour-nuke/blob/master/colour_nuke/notebooks/mmColorTarget_dependencies.ipynb
    # Source chromaticities are based on XRite ColorChecker 2005 values as described here:
    # https://github.com/colour-science/colour/blob/cdbffd063b0c44bb32d752b01647137871434851/colour/characterisation/dataset/colour_checkers/chromaticity_coordinates.py#L114-L114
    # http://www.babelcolor.com/colorchecker.htm
    # http://www.babelcolor.com/colorchecker-2.htm#CCP2_beforeVSafter
    # https://github.com/colour-science/colour-nuke/blob/master/colour_nuke/scripts/colour_rendition_chart.nk

    write_data(np.array(src_patches), node['data_label'].getValue())

    np_matrix = np.transpose(np.linalg.lstsq(np.array(src_patches), np.array(dst_patches))[0])
    matrix = np.ravel(np_matrix).tolist()
    node['matrix'].setValue(matrix)
    nuke.root().begin()


if __name__ == '__main__':
    calc_mtx()