# Command System

PoCA now uses one command model with three distinct roles:

- `CommandInfo`: persistent JSON input
- `CommandExecutionContext`: transient execution input
- `CommandExecutionResult`: transient execution output

## Intent

`CommandInfo` is the serializable envelope used for:

- macro recording and replay
- initialization files
- JSON-driven command creation
- user-requested persistent parameters

It should not be used to move runtime-only data during execution.

## Runtime Data

Use `CommandExecutionContext` for transient inputs such as:

- active camera
- export targets
- runtime-only object references

Use `CommandExecutionResult` for transient outputs such as:

- created objects
- generated meshes
- picking results
- temporary analysis handles

## JSON Creation

The preferred command creation path is schema-based:

1. define a `CommandSpec`
2. call `CommandSpec::create(...)`
3. let `CommandableObject::createCommand(...)` use the spec-first path

Custom `createCommand(...)` logic should only remain when the payload shape is
genuinely dynamic or delegated to another runtime owner.

## Serialized Parameter Types

When a command is saved through a `CommandSpec`, each known parameter is saved
with its schema type:

```json
{
  "clustersForChallenge": {
    "minNbLocs": { "type": "unsignedInteger", "value": 3 }
  }
}
```

Older files that store raw values are still accepted. Command creation unwraps
typed values before execution, so execution code continues to receive ordinary
JSON values.

Where the default value already carries the desired C++ type, prefer
`getParameterOr("minNbLocs", minNbLocs)` over
`getParameter<size_t>("minNbLocs")`. This lets the call site avoid repeating the
template argument while keeping typed local variables.

## Rule Of Thumb

- if it must survive serialization, it belongs in `CommandInfo`
- if it only exists during execution, it belongs in context or result
