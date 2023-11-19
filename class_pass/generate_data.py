import random

OUTFILE = 'data.txt'

sample_size = 1000000
max_hours = 10


def will_pass_class(*, num_hours: int, percent_lectures_attended: float) -> bool:
    lecture_normalization = int(percent_lectures_attended / 10)
    return lecture_normalization + num_hours > 12

def run():

    samples = []
    for i in range(sample_size):
        num_hours = random.randint(0, max_hours)
        percent_lectures_attended = random.random() * 100
        should_pass = will_pass_class(num_hours=num_hours, percent_lectures_attended=percent_lectures_attended)
        label = 1 if should_pass else 0
        samples.append((num_hours, percent_lectures_attended, label))

    with open(OUTFILE, 'w') as f:
        for sample in samples:
            f.write(f'{sample[0]},{sample[1]},{sample[2]}\n')


if __name__ == '__main__':
    run()