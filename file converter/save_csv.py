import pandas
import os
import argparse
import time


def main():
    parser = argparse.ArgumentParser()

    # Environment argument
    parser.add_argument('--mode', choices=['train', 'inference'], help='run mode')
    parser.add_argument('-cuda', action='store_true', help='Using GPU processor')
    parser.add_argument('-pin_memory', action='store_true', help='Load dataset while learning')
    parser.add_argument('-g', action='store_true', help='choose model between G and D')
    parser.add_argument('-d', action='store_true', help='choose model between G and D')
    parser.add_argument('--worker', type=int, default=1)
    parser.add_argument('--grey_scale', action='store_true', help='on grey scale image')

    # Inference parameter
    parser.add_argument('--inference_model_path', type=str,
                        default='')
    parser.add_argument('--input_size', type=int, default=228)
    parser.add_argument('--output_size', type=int, default=1)

    args = parser.parse_args()

    data_class = 'A'  # A or B
    data_dir = 'MY AWESOME CSV DIR' + data_class
    input_csv_file = 'MY AWESOME CSV DIR'
    output_csv_file = 'MY AWESOME CSV DIR'

    data_file_dir = os.listdir(data_dir)
    csv_file = pandas.read_csv(input_csv_file)

    for i, fn in enumerate(data_file_dir):
        image_dir = data_dir + os.sep + fn  # image or some fies...

        """
        some output you want to save in loop
        """
        output_a = 'a'
        output_b = 'b'

        csv_file.at[i, data_class] = output_a
        csv_file.at[i, data_class] = output_b

        print(fn + ' Done!!!')

    pandas.DataFrame.to_csv(csv_file, output_csv_file, index=False)  # save csv


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time : %f' % (time.time() - start_time))
