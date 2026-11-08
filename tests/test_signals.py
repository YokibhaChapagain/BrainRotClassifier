from brainrot.signals import analyze_video, extract_audio


def test_calm_video_has_few_cuts(calm_video):
    s, kfs = analyze_video(calm_video)
    assert 11 <= s.duration_s <= 13
    assert s.scene_cuts <= 1
    assert not s.is_vertical
    assert s.has_audio and s.mean_volume_db is not None
    assert kfs == []


def test_hyper_video_has_many_cuts(hyper_video):
    s, _ = analyze_video(hyper_video)
    assert s.is_vertical
    assert s.cuts_per_minute > 150  # cuts every 0.25s => ~240/min (some same-colour neighbours aside)


def test_keyframes_cover_whole_video(calm_video):
    s, kfs = analyze_video(calm_video, max_frames=6, extract_keyframes=True)
    assert len(kfs) == 6
    ts = [k.timestamp for k in kfs]
    assert ts == sorted(ts)
    assert ts[0] < 2.5 and ts[-1] > 9.5
    assert all(k.jpeg[:2] == b"\xff\xd8" for k in kfs)


def test_max_seconds_limits_analysis(calm_video):
    s, _ = analyze_video(calm_video, max_seconds=4)
    assert s.duration_s <= 4.5


def test_extract_audio(calm_video, tmp_path):
    out = extract_audio(calm_video, tmp_path / "a.mp3", max_seconds=5)
    assert out is not None and out.stat().st_size > 1000
