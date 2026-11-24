"""The studio's tools image: `mnm/studio-tools`, built on the box as ComfyUI's is (an official
base, every package pinned by sha256, the weights checked at build), recorded in
`<state>/studio/tools.json`, and run as one `docker run --rm --network none --read-only` per
job, as the user's uid, over one run folder mounted read-only (and, for a job that writes files,
one empty folder mounted writable at /out, whose files mnm moves into place after checking
each). Nothing is pulled and nothing runs until the user installs it (`mnm studio install`);
every studio step that needs it says so and goes on without it.

`mnm engines update` never touches it (it is built, not pulled); a new pin in mnm is a new tag,
and `mnm studio install` builds it, keeping the one before for `rollback`.
"""

from __future__ import annotations

import hashlib
import secrets
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from makenomistakes.comfy import build_stream
from makenomistakes.comfy.image import build_options
from makenomistakes.config import Settings
from makenomistakes.studio.built_image import (
    EXIT_WORDS,
    BuiltImage,
    ImageSpec,
    sandbox,
)
from makenomistakes.studio.keys import Key
from makenomistakes.studio.tools_answers import (
    MasterAnswer,
    MidiAnswer,
    ScoreAnswer,
    SimilarityAnswer,
    StructureMap,
    ToolsError,
    ToolsMissingError,
    TuneAnswer,
    answered_error,
    parse,
)

DOCKER_DIR = Path(__file__).parent / "docker"
IMAGE_REPO = "mnm/studio-tools"
IMAGE_GIB = 3.7  # torch for the CPU, librosa, numba and every tool's weights (as a build reported)
BUILD_SECONDS = 3600.0
JOB_SECONDS = 1800.0  # a 10-minute song's score or structure on a slow CPU
TMP_SIZE = "2g"  # numba's cache and the libraries' scratch; the root is read-only
OUT_PREFIX = ".tools-out-"  # a job's writable folder, inside the run's while it runs
JOB_PREFIX = "mnm-tools-"  # a job's container: removed by name when it outlasts its wait
JOB_LABEL = "mnm.instance=studio-tools"  # and found by the uninstall, as every mnm container
# The weights the image carries, by source and sha256 (the build checks each)
SCORER_REVISION = "9b1dd8e5df9af7216e836a98974fe3b82c56ded6"  # facebook/audiobox-aesthetics
SCORER_SHA256 = "a5a3c2412649cc2384ec525ffd5180ce6c4778f43bed6108e0a1303de04d014e"
SCORER_CONFIG_SHA256 = "0b8eabc5ced92cefed116a3aca8f1e59d2a33dc4d376b9e86dfb9c072e1d280d"
BEATS_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
BEATS_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"  # speechbrain/spkrec-ecapa-voxceleb
ECAPA_SHA256 = "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
BUILT_FROM = (
    "Dockerfile",
    "requirements-amd64.txt",
    "requirements-arm64.txt",
    "requirements-nodeps.txt",
    "mnm_tools.py",
    "mnm_tune.py",
)
# What the image carries, as the user reads it (`mnm studio status`, System → Studio)
CARRIES = (
    "Audiobox Aesthetics 0.0.4 (Meta, CC-BY-4.0): the candidates' score",
    "Beat This! 1.1.0 (CPJKU, MIT): beats, downbeats and the tempo",
    "librosa 1.0.0 (ISC): the parts of the song, by a novelty curve over its features",
    "SpeechBrain 1.1.2's ECAPA-TDNN (Apache-2.0): how alike a sung voice is to its reference",
    "Basic Pitch 0.4.0 (Spotify, Apache-2.0), its ONNX model: each stem's notes as MIDI",
    "Matchering 2.0.6 (GPL-3.0-or-later): a mix mastered to a reference track",
    "torchcrepe 0.0.24 (MIT), CREPE's weights in its wheel: the vocals tuned toward the key",
)
NOT_INSTALLED = (
    f"The studio tools are not installed: `mnm studio install` builds them (about "
    f"{IMAGE_GIB:.0f} GB, 10-20 minutes), or System → Studio."
)
SPEC = ImageSpec(
    repo=IMAGE_REPO,
    ledger="tools.json",
    lock=".tools.lock",
    label="studio tools",
    possessive="The studio tools'",
    install="mnm studio install",
    uninstall="mnm studio uninstall",
    not_installed=NOT_INSTALLED,
    gib=IMAGE_GIB,
    minutes="10-20",
    build_seconds=BUILD_SECONDS,
    error=ToolsError,
    missing=ToolsMissingError,
)


def recipe_hash() -> str:
    """What the image is built from: its files and the weights' pins."""
    pins = (
        SCORER_REVISION, SCORER_SHA256, SCORER_CONFIG_SHA256, BEATS_URL, BEATS_SHA256,
        ECAPA_REVISION, ECAPA_SHA256,
    )  # fmt: skip
    digest = hashlib.sha256("\n".join(pins).encode())
    for name in BUILT_FROM:
        digest.update((DOCKER_DIR / name).read_bytes())
    return digest.hexdigest()[:12]


def image_tag() -> str:
    return f"{IMAGE_REPO}:{recipe_hash()}"


class StudioTools(BuiltImage):
    """The tools image (`BuiltImage`: its ledger, build, rollback, removal) and its jobs."""

    spec = SPEC

    def image_tag(self) -> str:
        return image_tag()

    def build_argv(self, tag: str, ca: Path | None = None) -> list[str]:
        argv = [
            "docker", "build", build_stream.PLAIN, "--tag", tag,
            "--build-arg", f"SCORER_REVISION={SCORER_REVISION}",
            "--build-arg", f"SCORER_SHA256={SCORER_SHA256}",
            "--build-arg", f"SCORER_CONFIG_SHA256={SCORER_CONFIG_SHA256}",
            "--build-arg", f"BEATS_URL={BEATS_URL}",
            "--build-arg", f"BEATS_SHA256={BEATS_SHA256}",
            "--build-arg", f"ECAPA_REVISION={ECAPA_REVISION}",
            "--build-arg", f"ECAPA_SHA256={ECAPA_SHA256}",
            "--label", "mnm.engine=studio-tools",
        ]  # fmt: skip
        return [*argv, *build_options(ca), str(DOCKER_DIR)]

    # -- jobs ----------------------------------------------------------------------------

    def run_argv(
        self,
        image: str,
        folder: Path,
        command: str,
        *names: str,
        job: str,
        out: Path | None = None,
        words: tuple[str, ...] = (),
    ) -> list[str]:
        """`names` are files in `folder` (read as `/work/<name>`); `words` follow them as they
        are (tuning's key and two numbers), never a path. `job` names the container."""
        writes = ["-v", f"{out}:/out:rw"] if out is not None else []
        return [
            "docker", "run", "--rm", "--name", job, "--label", JOB_LABEL, *sandbox(TMP_SIZE),
            "-v", f"{folder}:/work:ro", *writes, image, command,
            *(f"/work/{_plain(name)}" for name in names), *words,
        ]  # fmt: skip

    def _job(
        self,
        folder: Path,
        command: str,
        *names: str,
        into: Path | None = None,
        words: tuple[str, ...] = (),
    ) -> str:
        ledger = self.installed()
        for name in names:
            if not (folder / _plain(name)).is_file():
                raise ToolsError(f"There is no {name} in {folder} to read.")
        job = f"{JOB_PREFIX}{secrets.token_hex(4)}"
        argv = self.run_argv(ledger.image, folder, command, *names, job=job, out=into, words=words)
        code, out, err = self.runner.run(argv, timeout=JOB_SECONDS)
        if code == 124:  # the docker client was stopped, never the container: stop it here
            gone, _out, _err = self.runner.run(["docker", "rm", "--force", job], timeout=60.0)
            left = "" if gone == 0 else f"; its container is still there: `docker rm --force {job}`"
            raise ToolsError(
                f"The studio tools' {command} took over {JOB_SECONDS / 60:.0f} min and was "
                f"stopped{left}."
            )
        if code != 0 and not answered_error(out):  # killed before its own error line
            why = EXIT_WORDS.get(code, f"exit {code}")
            tail = err.strip()[-300:]
            raise ToolsError(
                f"The studio tools' {command} stopped ({why})" + (f": {tail}" if tail else ".")
            )
        return out

    def score(self, folder: Path, name: str) -> ScoreAnswer:
        return parse(self._job(folder, "score", name), ScoreAnswer)

    def structure(self, folder: Path, name: str) -> StructureMap:
        return parse(self._job(folder, "structure", name), StructureMap)

    def similarity(self, folder: Path, reference: str, *names: str) -> SimilarityAnswer:
        """How alike each file's voice is to the reference's, all in one folder."""
        return parse(self._job(folder, "similarity", reference, *names), SimilarityAnswer)

    def midi(self, folder: Path, targets: dict[str, str]) -> MidiAnswer:
        """Each file's notes as a `.mid` in `folder` under the name mnm gives it
        (`targets`: the file read, then the name kept). The answer must name exactly the files
        sent, each with the `.mid` the job writes for it (`<its stem>.mid`): a name of the
        job's own is refused, never moved over a file of the song."""
        names = list(targets)
        with self._writing(folder) as out:
            answer = parse(self._job(folder, "midi", *names, into=out), MidiAnswer)
            if sorted(answer.files) != sorted(names):
                raise ToolsError(
                    f"The studio tools answered for {', '.join(sorted(answer.files)) or 'nothing'}"
                    f", not the files sent ({', '.join(names)})."
                )
            for name, target in targets.items():
                wrote = answer.files[name].midi
                _moved(out, wrote, folder, target, expected=f"{Path(name).stem}.mid")
        return answer

    def master(self, folder: Path, mix: str, reference: str, target: str) -> MasterAnswer:
        """The mix mastered to the reference as `target` in `folder`."""
        with self._writing(folder) as out:
            answer = parse(self._job(folder, "master", mix, reference, into=out), MasterAnswer)
            _moved(out, answer.file, folder, target, expected="master.wav")
        return answer

    def tune(
        self, folder: Path, source: str, target: str, key: Key, strength: int, speed: int
    ) -> TuneAnswer:
        """`source`'s voice moved toward `key`'s notes as `target` in `folder`."""
        mode = "minor" if key.minor else "major"
        with self._writing(folder) as out:
            words = (str(key.tonic), mode, str(strength), str(speed))
            answer = parse(self._job(folder, "tune", source, into=out, words=words), TuneAnswer)
            _moved(out, answer.file, folder, target, expected="vocals-tuned.wav")
        return answer

    @contextmanager
    def _writing(self, folder: Path) -> Iterator[Path]:
        """An empty folder beside `folder` for a job's files (the job runs as the user's uid,
        so it can write there); removed after, whatever the job left."""
        out = folder / f"{OUT_PREFIX}{secrets.token_hex(4)}"
        out.mkdir(mode=0o700)
        try:
            yield out
        finally:
            shutil.rmtree(out, ignore_errors=True)


def _moved(out: Path, name: str, folder: Path, target: str, expected: str) -> Path:
    """The file the job wrote in its out folder, moved into `folder` under mnm's own name
    (`target`): only the name the job was told to write (`expected`), and only a plain file
    directly there, never a link (the job could have written one pointing anywhere)."""
    if name != expected:
        raise ToolsError(f"The studio tools named {name} where they were to write {expected}.")
    source = out / _plain(name)
    if source.is_symlink() or not source.is_file():
        raise ToolsError(f"The studio tools named {name} and wrote no such file.")
    return source.replace(folder / _plain(target))


def _plain(name: str) -> str:
    """A file directly in the mounted folder: no path, no leading dash."""
    if "/" in name or name.startswith(("-", ".")) or not name:
        raise ToolsError(f"'{name}' is not a file name the studio tools are given.")
    return name


def studio_tools(settings: Settings) -> StudioTools:
    """This box's tools (tests put one with a fake runner in its place)."""
    return StudioTools(settings)
