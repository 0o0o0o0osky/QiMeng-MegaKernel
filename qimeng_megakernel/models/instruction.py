from dataclasses import dataclass, fields
from typing import Any, Callable


@dataclass
class Instruction:
    @classmethod
    def opcode(cls) -> int:
        raise NotImplementedError

    @classmethod
    def prev_opcode(cls) -> tuple[int, ...]:
        raise NotImplementedError

    def serialize(self):
        words = [self.opcode()]
        for field in fields(self):
            name = field.name
            attr = getattr(self, name)

            if isinstance(attr, int):
                words.append(attr)
            elif isinstance(attr, tuple):
                words.append(len(attr))
                words.extend(attr)
            elif isinstance(attr, list):
                words.append(len(attr))
                words.extend(attr)
            # for convenience
            elif attr is None:
                words.append(0)
            else:
                raise ValueError(f"Unsupported field type: {attr}")

        return words


def run_pyvm_sequence(
    globs: Any,
    instructions: list[Instruction],
    handler_map: dict[type[Instruction], Callable[[Any, Instruction], None]],
) -> None:
    """Run a sequence of instructions with the given handler map (VM executor). Used by decode, not verifier-only."""
    for ins in instructions:
        handler = handler_map.get(type(ins))
        if handler is None:
            raise KeyError(f"Missing solver for instruction: {type(ins).__name__}")
        handler(globs, ins)
