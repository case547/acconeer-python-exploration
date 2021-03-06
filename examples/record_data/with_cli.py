import os
import sys

import acconeer.exptool as et


def main():
    parser = et.utils.ExampleArgumentParser()
    parser.add_argument("-o", "--output-file", type=str, required=True)
    parser.add_argument("-l", "--limit-frames", type=int)
    args = parser.parse_args()
    et.utils.config_logging(args)

    if os.path.exists(args.output_file):
        print("File '{}' already exists, won't overwrite".format(args.output_file))
        sys.exit(1)

    _, ext = os.path.splitext(args.output_file)
    if ext.lower() not in [".h5", ".npz"]:
        print("Unknown format '{}'".format(ext))
        sys.exit(1)

    if args.limit_frames is not None and args.limit_frames < 1:
        print("Frame limit must be at least 1")
        sys.exit(1)

    if args.socket_addr:
        client = et.SocketClient(args.socket_addr)
    elif args.spi:
        client = et.SPIClient()
    else:
        port = args.serial_port or et.utils.autodetect_serial_port()
        client = et.UARTClient(port)

    config = et.configs.EnvelopeServiceConfig()
    config.sensor = args.sensors
    config.update_rate = 30

    session_info = client.setup_session(config)

    recorder = et.recording.Recorder(sensor_config=config, session_info=session_info)

    client.start_session()

    interrupt_handler = et.utils.ExampleInterruptHandler()
    print("Press Ctrl-C to end session")

    i = 0
    while not interrupt_handler.got_signal:
        data_info, data = client.get_next()
        recorder.sample(data_info, data)

        i += 1

        if args.limit_frames:
            print("Sampled {:>4}/{}".format(i, args.limit_frames), end="\r", flush=True)

            if i >= args.limit_frames:
                break
        else:
            print("Sampled {:>4}".format(i), end="\r", flush=True)

    print()

    client.disconnect()

    record = recorder.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    et.recording.save(args.output_file, record)
    print("Saved to '{}'".format(args.output_file))


if __name__ == "__main__":
    main()
