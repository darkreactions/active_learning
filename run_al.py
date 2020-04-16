import argparse
from projects import leave_one_out, phase_mapping

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs selected project')
    parser.add_argument('project', metavar='P', type=str, nargs='+',
                        help='Project names to run')

    args = parser.parse_args()

    for project in args.project:
        if project == 'leave_one_out':
            leave_one_out.run()
        elif project == 'phase_mapping':
            phase_mapping.run()
