// ---------------------------------------------------------------------------------------
//                                        ILGPU
//                        Copyright (c) 2016-2020 Marcel Koester
//                                    www.ilgpu.net
//
// File: ExchangeBuffer.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details
// ---------------------------------------------------------------------------------------

using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace ILGPU.Runtime
{
    /// <summary>
    /// Specifies the allocation mode for a single exchange buffer.
    /// </summary>
    public enum ExchangeBufferMode
    {
        /// <summary>
        /// Prefer page locked memory for improved transfer speeds.
        /// </summary>
        PreferPageLockedMemory = 0,

        /// <summary>
        /// Allocate CPU memory in pageable memory to leverage virtual memory.
        /// </summary>
        UsePageablememory = 1,
    }

    /// <summary>
    /// Represents an opaque memory buffer that contains a GPU and a CPU back buffer.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <remarks>Members of this class are not thread safe.</remarks>
    public class ExchangeBuffer<T> : AcceleratorObject
        where T : unmanaged
    {
        #region Instance

        /// <summary>
        /// Initializes a new basic exchange buffer.
        /// </summary>
        /// <param name="gpuBuffer">The parent GPU buffer to use.</param>
        /// <param name="mode">The exchange buffer mode to use.</param>
        /// <remarks>
        /// CAUTION: The ownership of the <paramref name="gpuBuffer"/> is transfered to
        /// this instance.
        /// </remarks>
        protected internal ExchangeBuffer(
            MemoryBuffer<ArrayView1D<T, Stride1D.Dense>> gpuBuffer,
            ExchangeBufferMode mode)
            : base(gpuBuffer.Accelerator)
        {
            GPUBuffer = gpuBuffer;
            CPUBuffer = Accelerator is CudaAccelerator &&
                mode == ExchangeBufferMode.PreferPageLockedMemory
                ? CPUMemoryBuffer.CreatePinned(
                    Accelerator,
                    gpuBuffer.LengthInBytes,
                    gpuBuffer.ElementSize)
                : CPUMemoryBuffer.Create(gpuBuffer.LengthInBytes, gpuBuffer.ElementSize);
            CPUView = new ArrayView<T>(CPUBuffer, 0, gpuBuffer.Length);
        }

        #endregion

        #region Properties

        /// <summary>
        /// Returns the owned and underlying CPU buffer.
        /// </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        protected CPUMemoryBuffer CPUBuffer { get; }

        /// <summary>
        /// Returns the owned and underlying GPU buffer.
        /// </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        protected MemoryBuffer<ArrayView1D<T, Stride1D.Dense>> GPUBuffer { get; }

        /// <summary>
        /// The CPU view representing the allocated chunk of memory.
        /// </summary>
        public ArrayView<T> CPUView { get; }

        /// <summary>
        /// The GPU view representing the allocated chunk of memory.
        /// </summary>
        public ArrayView<T> GPUView => GPUBuffer.View;

        /// <summary>
        /// Returns the length of this array view.
        /// </summary>
        public long Length => GPUBuffer.Length;

        /// <summary>
        /// Returns the element size.
        /// </summary>
        public int ElementSize => GPUBuffer.ElementSize;

        /// <summary>
        /// Returns the length of this buffer in bytes.
        /// </summary>
        public long LengthInBytes => GPUBuffer.LengthInBytes;

        /// <summary>
        /// Returns a span pointing to the CPU part of the buffer.
        /// </summary>
        public unsafe Span<T> Span
        {
            get
            {

                IndexTypeExtensions.AssertIntIndexRange(Length);
                return new Span<T>(
                    Unsafe.AsPointer(ref CPUView.LoadEffectiveAddress()),
                    (int)Length);
            }
        }

        /// <summary>
        /// Returns a reference to the i-th element on the CPU.
        /// </summary>
        /// <param name="index">The element index.</param>
        /// <returns>A reference to the i-th element on the CPU.</returns>
        public ref T this[Index1D index] => ref CPUView[index];

        /// <summary>
        /// Returns a reference to the i-th element on the CPU.
        /// </summary>
        /// <param name="index">The element index.</param>
        /// <returns>A reference to the i-th element on the CPU.</returns>
        public ref T this[LongIndex1D index] => ref CPUView[index];

        #endregion

        #region Methods

        /// <summary>
        /// Copes data from CPU memory to the associated accelerator.
        /// </summary>
        public void CopyToAccelerator() => CopyToAccelerator(Accelerator.DefaultStream);

        /// <summary>
        /// Copies data from CPU memory to the associated accelerator.
        /// </summary>
        /// <param name="stream">The stream to use.</param>
        public void CopyToAccelerator(AcceleratorStream stream) =>
            CopyToAccelerator(stream, 0L, Length);

        /// <summary>
        /// Copies data from CPU memory to the associated accelerator.
        /// </summary>
        /// <param name="offset">The target memory offset.</param>
        /// <param name="length">The length (number of elements).</param>
        public void CopyToAccelerator(long offset, long length) =>
            CopyToAccelerator(Accelerator.DefaultStream, offset, length);

        /// <summary>
        /// Copies data from CPU memory to the associated accelerator.
        /// </summary>
        /// <param name="stream">The stream to use.</param>
        /// <param name="offset">The element offset.</param>
        /// <param name="length">The length (number of elements).</param>
        public void CopyToAccelerator(
            AcceleratorStream stream,
            long offset,
            long length)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            var sourceView = CPUView.SubView(offset, length);
            var targetView = GPUView.SubView(offset, length);

            sourceView.CopyTo(stream, targetView);
        }

        /// <summary>
        /// Copies data from the associated accelerator into CPU memory.
        /// </summary>
        public void CopyFromAccelerator() =>
            CopyFromAccelerator(Accelerator.DefaultStream);

        /// <summary>
        /// Copies data from the associated accelerator into CPU memory.
        /// </summary>
        /// <param name="stream">The stream to use.</param>
        public void CopyFromAccelerator(AcceleratorStream stream) =>
            CopyFromAccelerator(stream, 0L, Length);

        /// <summary>
        /// Copies data from the associated accelerator into CPU memory.
        /// </summary>
        /// <param name="offset">The element offset.</param>
        /// <param name="length">The length (number of elements).</param>
        public void CopyFromAccelerator(long offset, long length) =>
            CopyFromAccelerator(Accelerator.DefaultStream, offset, length);

        /// <summary>
        /// Copies data from the associated accelerator into CPU memory.
        /// </summary>
        /// <param name="stream">The stream to use.</param>
        /// <param name="offset">The element offset.</param>
        /// <param name="length">The length (number of elements).</param>
        public void CopyFromAccelerator(
            AcceleratorStream stream,
            long offset,
            long length)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            var sourceView = GPUView.SubView(offset, length);
            var targetView = CPUView.SubView(offset, length);
            sourceView.CopyTo(stream, targetView);
        }

        #endregion

        #region IDisposable

        /// <summary>
        /// Frees the underlying CPU and GPU memory handles.
        /// </summary>
        protected override void DisposeAcceleratorObject(bool disposing)
        {
            if (!disposing)
                return;

            CPUBuffer.Dispose();
            GPUBuffer.Dispose();
        }

        #endregion
    }
}
