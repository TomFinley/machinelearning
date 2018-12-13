# Internalization

We must draw a distinction between two things: the public surface of the API
that users encounter when trying to use ML.NET, and the infrastructure by
which authors of components implement usable components in ML.NET.

To explain what I mean, I will take the most ubiquitous and important example
in our codebase: we have of course `IDataView`, yet, if you look through the
implementation og ML.NET fully articulated implementations of `IDataView` from
absolute scratch are relatively rare. Instead, we have common patterns that
are implemented using a jungle of simpler-to-implement interfaces and base
classes, from which `IDataView`s are derived. For example: when scoring, say,
using a linear model, there is a relatively simpler interface `IValueMapper`
or `IValueMapperDist` that describes what sort of input it will accept (a
vector of floats of a particular dimension), the output it will produce (a
floating point score, and in the case of a calibrated predictor the
probability score). Now from a user's point of view, this looks like a regular
`ITransformer`, but in reality under the hood this is being mapped from a very
specialized interface `IValueMapper`, from which the crucial public interface
implementations `ITransformer`, `IRowToRowMapper`, and `IDataView` can be
derived.

However, is there any real reason to have people be able to implement
`IValueMapper` as an accessible public type? Maybe. Exposing this interface
would ease the burden of people writing their own implementations of
`ITransformer`. But for people that just want to use these base public types
of `ITransformer`, `IRowToRowMapper`, and `IDataView`, there is no purpose.

Also, since we have some limited time in which to "stabilize" the public API,
we have decided that it is efficacious to hide these sort of "infrastructure"
classes as much as possible, since the alternative is that we agree that the
shape of what is currently public is what we want to have now and forever --
which we most certainly do not want to do. We have for years had the attitude
that things that *can* be `internal` should be `internal`, but we've recently
expanded this considerably to allow cross-assembly "friend" references to
restrict things even further, so as to distinguish between things we want
accessible outside of an assembly by *our* code and by *user* code, pursuant
to [this](https://github.com/dotnet/machinelearning/pull/1520) and other
related work.

So much for the motivation for internalization. Assuming we count that as
settled, in the process of performing internalization several common questions
have come up, about how best to achieve internalization. This will necessary
touch on some features of C# that are not, evidently, things people encounter
every day, but that are nonetheless needful to know.

## Basic Internalization

Under normal circumstances, we previously would have made any accessible
members on the class `public`, and any members intended to be used within
subclasses as `protected`. Semi-recently, C# 7.2 introduced the new combined
modifier [`private
protected`](https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/keywords/private-protected).

You can effectively make a base class uninstantiable by any class by making
its base constructor `private protected` (possibly with a `BestFriend`
attribute), all extension methods likewise `private protected`.

Note that this is important: members that are `protected` are definitely part
of the surface of the API!

## De-interfacing

When you declare an interface public, that means that all the implementation
is (one way or the other) public as well. There is no such thing as a *public*
interface over an internal structure; if you have a public type implementing a
public interface, everything on it becomes part of your public surface area.

One "simple" way to overcome this problem is to just not use an interface;
that is, instead of having there be an interface, instead use a base class.
This is also more in line with .NET recommendations. Sometimes, this is simply
impossible; interfaces allow some multiple inheritance relationships and
co/contra-variance that are simply impossible with classes. But, sometimes it
is possible, and we should always look for opportunities to turn interfaces
into classes, if we can.

Once it is not an interface, you can make whatever public surface we want
public, and whatever internal infrastructure we want `internal` or `private
protected`, as appropriate.

## Explicit Interface Implementations

For various reasons, merely turning everything into a class instead of an
interface is impossible. Interfaces have some basic capabilities around
covariance/contravariance and multiple inheritance and other such things, that
classes simply cannot achieve!

Let's imagine that we make an interface internal, so that the ML.NET code
itself is aware that a type implements an interface, but user code needn't be
concerned with this detail. (Even though the fact that it implements this
interface may still be important to the internal infrastructure.) So consider
this hypothetical interface `IFoo` that deals with the class `NonPublicThing`,
and the implementing class `Foo`.

```csharp
public interface IFoo
{
    NonPublicThing Bar { get; }
    NonPublicThing Biz(NonPublicThing blam);
}

public sealed class NonPublicThing { }

public sealed class Foo : IFoo
{
    public NonPublicThing Bar => ...;
    public NonPublicThing Biz(NonPublicThing blam) => ...;
}
```

Now let's imagine that we want to make `NonPublicThing` `internal`. That's
fine. This also usually implies we also want to make `IFoo` internal as well.
The way that we can make the *public* class implement this *internal*
interface in a way that is hidden is via an explicit interface.

```csharp
internal interface IFoo
{
    NonPublicThing Bar { get; }
    NonPublicThing Biz(NonPublicThing blam);
}

internal sealed class NonPublicThing { }

public sealed class Foo : IFoo
{
    NonPublicThing IFoo.Bar => ...;
    NonPublicThing IFoo.Biz(NonPublicThing blam) => ...;
}
```

Note that henceforth an instance of `Foo` will not, externally, appear to
implement `IFoo`, and even internally an instance typed `Foo` will not have
its `Bar` or `Biz` members visible. However by casting or assigning to an
`IFoo` typed value, then these methods *will* become visible and usable.

So now, the class still implements `IFoo`, but that detail, including the
implementing methods, is hidden from the user.

Note that this also implies that if it is necessary for a user to perform some
operation that depends on this interface (which is presumably an interface
only because its implementation on this type of object is optional), then the
check to see whether the operation is actually applicable could only be a
runtime check. Sometimes this is perfeectly acceptable. Yet, sometimes, it may
not be, which brings us to the next strategy:

## Secret Service

Sometimes merely making interfaces internal is insufficient. Often we most
definitely *want* to communicate the information that a certain type of object
is capable of performing this or that operation (e.g., for compile time
checks), and for various reasons as communicated earlier it must be done
through an interface and not via a base class, but we may not be anxious to
publicize the details of how the operation is performed.

So, to take an example, we might say, "this is a model capable of extracting
feature importance" or "this model can save itself to this format." But,
while making clear through the type system that it *can* do these things, the
nit and grit of how that operation is actually performed we may not want to
show. (Mostly, because we want to be free to change it in the future! As soon
as something is part of our public API surface, that's it.)

One strategy for dealing with these dual purposes is to have dual types. Note
the introduction of these intermediating `IFooable`/`Fooer` types.

```csharp
public interface IFooable
{
    Fooer Fooer { get; }
}

internal interface IFoo
{
    NonPublicThing Bar { get; }
    NonPublicThing Biz(NonPublicThing blam);
}

public sealed class Fooer
{
    internal IFoo Foo { get; };
    internal Fooer(IFoo foo) => Foo = foo;
}

internal sealed class NonPublicThing { }

public sealed class Foo : IFooable, IFoo
{
    Fooer IFooable.Fooer => new Fooer(this);

    NonPublicThing IFoo.Bar => ...;
    NonPublicThing IFoo.Biz(NonPublicThing blam) => ...;
}
```

This on the first glance appears to be more complex, but from the point of
view of the public surface, what have we revealed? Only this:

```csharp
public interface IFooable
{
    Fooer Fooer { get; }
}

public sealed class Fooer
{
}

public sealed class Foo : IFooable
{
    Fooer IFooable.Fooer => new Fooer(this);
}
```

We have revealed that `Foo` is capable of performing *some* operation through
this `IFooable`/`Fooer` thing (which is what we wanted to reveal), but the nit
and grit of how it works through `IFoo` and perhaps other calls through
`NonPublicThing` are used to accomplish these aims is still rendered
completely invisible through the intermediation of this wrapping `Fooer`
object.
