# One managed compute platform

> **Provenance class: calibrated reference.** This file is not part of the static
> Human Prior in this release. Load it only through a separately digested,
> provenance-bearing calibrated-note binding whose platform identity also appears
> in the capability lock and Stage Context Packet.

This platform offers batch jobs and interactive instances against a shared quota, driven
through a vendor CLI. A scheduler with submit directives, a container orchestrator, or
machines held directly work differently enough that little of the detail here carries over.

## What platforms differ on

Every platform has an answer to each of these, and the answers rarely resemble each other:

- **how work is described** — a JSON body, a submit script with directives, a manifest, a
  command line. Whatever it is, something in it is mandatory and undocumented.
- **how you are charged** — a reserved pool, an on-demand queue, a fair-share priority. This
  decides whether waiting is a thing that happens to you or a thing you can influence.
- **how the filesystem appears** — and in particular whether it appears identically to every
  way of running. Where it does not, path translation becomes a step that can be wrong.
- **what the runtime environment actually contains** versus what its description claims. The
  gap between the two is where multi-device work breaks.
- **whether interactive access exists at all**, and if so what it costs and how long it lasts.
- **how state is preserved between sessions** — a captured image, a configuration script, a
  home directory that persists. This decides how expensive it is to start again.

**Account identifiers live in a local configuration file, not here.** Everything below is
parameterised. See *What your configuration must provide*.

---

## The two tiers, concretely

**Batch jobs** — submitted with a full specification, scheduled against the quota, run to
completion. Every submission pays image pull and environment preparation before your first
line executes, on the order of minutes. Use for validated long runs.

**Interactive instances** — a held machine you connect into over the network. Iteration is
seconds. Use for everything up to the point the code runs clean.

The intended flow: create a temporary instance, connect in, iterate until training runs
clean, submit the validated version as a batch job, then release the instance. The instance
existing while a long job runs is pure cost.

**Instances have a hard lifetime here**, on the order of hours rather than days. Stopping and
restarting before expiry resets the timer; letting it expire does not. Put the expiry
somewhere visible — this is a common way to lose a debugging session.

---

## Path remapping between tiers

The shared filesystem is not mounted identically in both tiers. On the interactive side two
paths commonly resolve to the same storage; in a batch job only one of them exists.

**Every path in a submitted command must use the batch-side prefix.** Code that ran
interactively will fail on submission with a missing path, and the failure arrives after the
start-up cost rather than immediately.

Translate paths as an explicit step before submitting, and treat that translation as
something that can itself be wrong.

---

## The submitted command runs under a different shell

The default shell for a submitted command is not the interactive one. Shell builtins you rely
on — `source` in particular — are unavailable, so anything that activates an environment fails
immediately.

**Wrap the whole command in an explicit `bash -c '...'`.** Then, inside it, re-establish the
environment from scratch: source the environment manager's profile script, activate the
environment, change directory, and only then run the command. Nothing from a login shell is
present.

```bash
bash -c '. <conda_profile_script> && conda activate <env> && <library-path-fix> && cd <workdir> && <command>'
```

If the environment's location is unknown, probe for it on the shared filesystem rather than
guessing:

```bash
find <shared-mount> -maxdepth 5 -name "conda.sh" -path "*/etc/profile.d/*" 2>/dev/null
```

---

## The base image's runtime is older than the wheels expect

Container base images commonly ship a CUDA runtime older than the one the installed wheels
were built against. Single-GPU work often survives this; **multi-GPU work fails with an
undefined-symbol error** naming a driver entry point, which reads as a code problem and is not.

The fix is to put the pip-installed vendor libraries ahead of the system ones before launching:

```bash
NVLIBS=$(python -c "import site,glob,os; print(':'.join(sorted({os.path.dirname(p) for p in glob.glob(site.getsitepackages()[0]+'/nvidia/*/lib/*.so*')})))")
export LD_LIBRARY_PATH="$NVLIBS:$LD_LIBRARY_PATH"
```

Note the escaping when this is nested inside the `bash -c '...'` wrapper — the inner quotes
must be escaped, and getting this wrong produces an empty path variable and the original
error, with nothing indicating the substitution failed. Echo the variable once during a smoke
run.

---

## Request fields that are not optional

**Prepaid quota requires a resource-configuration block, not an instance-type field.** Using
the instance-type form against prepaid quota is rejected in a way that does not obviously name
the cause.

**Shared memory must be requested explicitly** on this class of resource. The default is small
enough that data loaders fail in ways that look like data problems.

**Mounting the shared filesystem requires the network configuration** — the VPC, security
group and switch — in the same request. Without it, the job is created and the mount is
absent.

**Use direct storage URIs, not registered dataset identifiers.** The dataset form fails on
this cluster with mount-point errors inside the VPC. This is the kind of thing that looks like
a permissions problem for an hour.

CPU and memory scale with GPU count in fixed proportion; take the platform's published ratio
rather than inventing one, and keep shared memory large.

**Queue time scales with what you request.** Ask for one or two accelerators when testing.

---

## Operations you will need

Both tiers expose the same shape: create/submit, list, get status, fetch logs, stop, delete.
For interactive instances there are additionally a lifecycle-event query, a start (to restart
a stopped instance and reset its timer), and command execution over the network.

Practical notes on the ones that behave unexpectedly:

- **Listing may return nothing while the resource exists.** Query by identifier directly
  before concluding anything is gone.
- **A queuing instance cannot be stopped, only deleted.** Stop is only available once running.
- **Logs are fetched per pod**, with the pod identifier derived from the job identifier and the
  role. A job with no logs is usually still in environment preparation, not failing.
- **Connecting to a running instance** works over the internal network by pod address, without
  port forwarding, provided the public key was supplied at creation. Fetch the address from
  the instance's detail response rather than assuming it is stable.

---

## Capturing a working environment

Once an interactive instance holds an environment that works, snapshot it to a registry image
rather than reconstructing it later. The operation is asynchronous and can take a long time
depending on size; it moves through commit and push phases before completing.

Constraints worth knowing before you start: the target registry must be reachable from the
same network as the instance; the snapshot name is used as the image tag and so is restricted
to lowercase letters, digits and hyphens; and paths can be excluded from the capture, which is
worth doing for anything large and reproducible.

Once complete, the resulting image is usable for both new instances and batch jobs — which is
what closes the loop between the two tiers.

---

## What your configuration must provide

Keep these outside the deliverable, in a local file the agent reads:

| | what it identifies |
|---|---|
| region, profile | which account and endpoint the CLI acts against |
| workspace | the container everything is created in |
| quota / resource id | which pool the work is charged to |
| instance type | the shape of machine the quota provides |
| storage URI and mount path | where the shared filesystem is and where it appears |
| VPC, security group, switch | the network the mount and connections require |
| registry base, namespace | where captured images are pushed |
| username subdirectory | your own area of the shared filesystem |

Ask for anything missing rather than guessing. A wrong identifier here produces a request that
is accepted and does the wrong thing, or a permissions error that reads as a bug.

---

## Checking these still hold

1. **Re-run a minimal submitted job after any platform or image change** — one accelerator,
   trivial command, printing the mounted paths and the library path. It exercises every trap
   above in a couple of minutes.
2. **Re-check the path mapping between tiers** rather than assuming it. It is configuration,
   and configuration changes without announcement.
3. **Confirm the library-path fix is still necessary** before carrying it into a new image; a
   fix that has become unnecessary is a source of confusion later.
4. **Verify identifiers against the configuration file** at the start of a session rather than
   at first use. A stale quota or workspace surfaces as a scheduling failure much later.
