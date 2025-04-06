from configs.app_config import AppConfig
from core.trackers_factory import TrackersFactory


def main():
    trackers = TrackersFactory.createTrackers(AppConfig)

    trackers.trackFace()

if __name__ == "__main__":
    main()