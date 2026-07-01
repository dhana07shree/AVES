# from src.pipeline import VideoPipeline
# import src.config as config


# def main():

#     pipeline = VideoPipeline(config.DEFAULT_VIDEO)

#     pipeline.run()


# if __name__ == "__main__":

#     main()






from src.pipeline import VideoPipeline
import src.config as config


def main():
    pipeline = VideoPipeline(config.DEFAULT_VIDEO)
    pipeline.run()


if __name__ == "__main__":
    main()